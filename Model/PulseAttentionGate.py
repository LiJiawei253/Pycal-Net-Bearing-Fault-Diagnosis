"""
Pulse Attention Gate (PAG)

该模块在特征序列上生成物理注意力掩码，并对输入特征进行门控：
  F_cal = F_raw ⊙ M_phy

当前实现包含两条分支：
- **Impulsiveness (Kurtosis)**: 多尺度局部峭度（滑窗统计）→ Sigmoid 映射
- **Periodicity (Local correlation)**: 多延迟局部相关（滑窗统计）→ 标准化 → Sigmoid 映射

默认采用乘法融合（强调两者同时高）：
  M_phy = M_imp ⊙ M_cyc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PulseAttentionGate(nn.Module):
    """
    Implementation of Pulse Attention Gate using local kurtosis computation
    with convolution approximation for efficient processing.
    
    公式: 
    y = x ⊙ σ(Conv1D(x^4, W_quad) / (Conv1D(x^2, W_sq) + ϵ)^2 - threshold)
    
    其中 W_sq 和 W_quad 是全1卷积核，⊙ 是逐元素相乘, σ 是sigmoid函数
    
    改进: 
    1. 加入可学习的threshold参数
    2. 引入周期性约束（循环自相关）增强物理信息
    3. 使用多尺度峭度计算，提高对不同周期故障信号的适应性
    """
    def __init__(self, kernel_sizes=[15, 31, 63], threshold = 3.0, epsilon=1e-6,
                 center_moments: bool = True,
                 use_cycle_weight: bool = True,
                 fixed_alpha: Optional[float] = None,
                 cycle_window: int = 63,
                 cycle_lags: Optional[list] = None,
                 cycle_gain: float = 10.0,  # 增加默认值从5.0到10.0以增强对比度
                 kurtosis_gain: float = 2.0,  # 新增：kurtosis attention的增益参数
                 use_multiplicative_fusion: bool = True):  # 新增：是否使用乘法融合
        super(PulseAttentionGate, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.epsilon = epsilon
        self.center_moments = center_moments
        self.use_cycle_weight = use_cycle_weight
        self.fixed_alpha = fixed_alpha
        self.cycle_window = cycle_window
        self.cycle_lags = cycle_lags
        self.cycle_gain = cycle_gain
        self.kurtosis_gain = kurtosis_gain  # 新增：kurtosis attention的增益
        self.use_multiplicative_fusion = use_multiplicative_fusion  # 新增：是否使用乘法融合
        
        # 创建多个尺度的全1卷积核，作为滑动窗口
        self.kernels_sq = nn.ModuleList()
        self.kernels_quad = nn.ModuleList()
        self.paddings = []
        
        for k_size in kernel_sizes:
            # 每个尺度的填充值
            padding = (k_size - 1) // 2
            self.paddings.append(padding)
            
            # 注册每个尺度的卷积核
            self.register_buffer(f'kernel_sq_{k_size}', torch.ones(1, 1, k_size))
            self.register_buffer(f'kernel_quad_{k_size}', torch.ones(1, 1, k_size))
        
        # 将threshold设为可学习参数，能够根据数据特性自适应调整
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
        # 新增：可学习门控参数
        self.gate_weight = nn.Parameter(torch.randn(1, 2))  # W_alpha
        self.gate_bias = nn.Parameter(torch.zeros(1))       # b_alpha
        
    def forward(self, x, return_mask: bool = False):
        """
        Args:
            x: 输入张量，形状为 [B, C, L]
               B=批次大小，C=通道数，L=信号长度
        Returns:
            output: 注意力门控后的输出，形状与输入相同 [B, C, L]
        """
        # 计算x^2和x^4（若采用中心矩，则稍后在每个尺度上对 (x - 局部均值) 计算）
        x_squared = x.pow(2)  # [B, C, L]
        x_quad = x.pow(4)     # [B, C, L]

        # 获取批次大小和通道数
        batch_size, channels, seq_len = x.shape
        
        # 在不同尺度上计算局部峭度
        kurtosis_list = []
        
        for i, k_size in enumerate(self.kernel_sizes):
            kernel_sq = getattr(self, f'kernel_sq_{k_size}')
            kernel_quad = getattr(self, f'kernel_quad_{k_size}')
            padding = self.paddings[i]
            
            if channels > 1:
                kernel_expanded = kernel_sq.repeat(channels, 1, 1)  # ones kernel for sums/means
                if self.center_moments:
                    local_mean = F.conv1d(
                        x,
                        kernel_expanded,
                        padding=padding,
                        groups=channels
                    ) / k_size
                    x_centered = x - local_mean
                    m2 = F.conv1d(x_centered.pow(2), kernel_expanded, padding=padding, groups=channels) / k_size + self.epsilon
                    m4 = F.conv1d(x_centered.pow(4), kernel_expanded, padding=padding, groups=channels) / k_size
                    numerator = m4
                    denominator = m2
                else:
                    numerator = F.conv1d(x_quad, kernel_expanded, padding=padding, groups=channels)
                    denominator = F.conv1d(x_squared, kernel_expanded, padding=padding, groups=channels) + self.epsilon
                local_kurtosis = numerator / denominator.pow(2)
            else:
                if self.center_moments:
                    local_mean = F.conv1d(x, kernel_sq, padding=padding) / k_size
                    x_centered = x - local_mean
                    m2 = F.conv1d(x_centered.pow(2), kernel_sq, padding=padding) / k_size + self.epsilon
                    m4 = F.conv1d(x_centered.pow(4), kernel_sq, padding=padding) / k_size
                    local_kurtosis = m4 / m2.pow(2)
                else:
                    numerator = F.conv1d(x_quad, kernel_quad, padding=padding)
                    denominator = F.conv1d(x_squared, kernel_sq, padding=padding) + self.epsilon
                    local_kurtosis = (numerator / denominator.pow(2)) * k_size
           
           

            kurtosis_list.append(local_kurtosis)
        
        
        local_kurtosis = torch.mean(torch.stack(kurtosis_list), dim=0)
        
        # 与阈值比较并通过sigmoid获得基于峭度的注意力权重
        # 改进：使用kurtosis_gain增强对比度，避免输出集中在0.5附近
        kurtosis_attention = torch.sigmoid(self.kurtosis_gain * (local_kurtosis - self.threshold))
     
        
        # 周期性约束：改为短时局部相关（更随时间变化）
        # 若未指定 lags，则从核大小中派生一组代表性延迟
        cycle_weight = None
        if self.use_cycle_weight:
            lags = self.cycle_lags
            if lags is None:
                # 取各尺度的 1/4 周期作为候选延迟，去重且 >=1
                lags = sorted(list({max(1, k // 4) for k in self.kernel_sizes}))
            k = max(3, int(self.cycle_window))
            padding = k // 2
            # 局部能量归一化
            local_energy = F.avg_pool1d(x.pow(2), kernel_size=k, stride=1, padding=padding) + self.epsilon
            corr_list = []
            for d in lags:
                # 向右平移 d: 对齐长度
                shifted = F.pad(x, (d, 0))[:, :, :-d]
                prod = x * shifted
                local_corr = F.avg_pool1d(prod, kernel_size=k, stride=1, padding=padding)
                r = local_corr / local_energy
                corr_list.append(r)
            # 取绝对值的最大相关作为周期强度
            cycle_strength = torch.stack([c.abs() for c in corr_list], dim=0).amax(dim=0)
            # 改进：使用标准化而不是去均值，增强对比度
            cycle_mean = cycle_strength.mean(dim=-1, keepdim=True)
            cycle_std = cycle_strength.std(dim=-1, keepdim=True) + self.epsilon
            z_normalized = (cycle_strength - cycle_mean) / cycle_std  # 标准化
            cycle_weight = torch.sigmoid(self.cycle_gain * z_normalized)
        
        # 融合方式：默认使用乘法融合（强调两者都高），也可选择可学习门控
        # kurtosis_attention, cycle_weight: [B, C, L]
        if self.use_cycle_weight and cycle_weight is not None:
            if self.use_multiplicative_fusion:
                # 改进：使用乘法融合，强调两者都高的情况，增加对比度
                # 注意：这替换了原来的可学习门控，但可学习门控代码仍保留在else分支中
                attention_weights = kurtosis_attention * cycle_weight
            else:
                # 原始加法融合方式（包括可学习门控）
                if self.fixed_alpha is not None:
                    alpha = torch.tensor(self.fixed_alpha, device=x.device).clamp(0.0, 1.0)
                    attention_weights = alpha * kurtosis_attention + (1 - alpha) * cycle_weight
                else:
                    # 可学习门控：使用gate_weight和gate_bias动态计算alpha
                    concat = torch.stack([kurtosis_attention, cycle_weight], dim=-1)  # [B, C, L, 2]
                    alpha = torch.sigmoid(torch.matmul(concat, self.gate_weight.t()) + self.gate_bias)  # [B, C, L, 1]
                    alpha = alpha.squeeze(-1)  # [B, C, L]
                    attention_weights = alpha * kurtosis_attention + (1 - alpha) * cycle_weight
        else:
            attention_weights = kurtosis_attention
        
        # 保存局部峭度和注意力权重，供钩子函数访问
        self.last_kurtosis = local_kurtosis.detach()
        self.last_attention = attention_weights.detach()
        self.last_kurtosis_attention = kurtosis_attention.detach()
        self.last_cycle_weight = cycle_weight.detach() if cycle_weight is not None else None
        self.last_multiscale_kurtosis = {f'scale_{k}': kurt.detach() for k, kurt in zip(self.kernel_sizes, kurtosis_list)}
        
        # 将注意力权重应用到原始输入
        enhanced = x * attention_weights
        if return_mask:
            return enhanced, attention_weights
        return enhanced








