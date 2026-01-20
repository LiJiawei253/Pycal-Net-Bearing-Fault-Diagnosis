import math

import torch
import torch.nn as nn
from Model.ConvQuadraticOperation import ConvQuadraticOperation
from Model.PulseAttentionGate import PulseAttentionGate

def hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.

        Parameters:
        -------------
        x (Tensor) : The signal data.
                     The Hilbert transform is performed along last dimension of `x`.

        Returns:
        -------------
        analytic (Tensor): A complex tensor with the same shape of `x`,
                           representing its analytic signal.

    '''

    N = x.shape[-1]
    # Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        x[..., 1: N // 2] *= 2
        x[..., N // 2 + 1:] = 0
    else:
        x[..., 1: (N + 1) // 2] *= 2
        x[..., (N + 1) // 2:] = 0
    return torch.fft.ifft(x)


def get_envelope_frequency(x, fs, ret_analytic=False, **kwargs):
    '''
        Compute the envelope and instantaneous freqency function of the given signal, using Hilbert transform.
        The transformation is done along the last axis.

        Parameters:
        -------------
        x (Tensor) :
            Signal data. The last dimension of `x` is considered as the temporal dimension.
        fs (real) :
            Sampling frequencies in Hz.
        ret_analytic (bool, optional) :
            Whether to return the analytic signal.
            ( Default: False )

        Returns:
        -------------
        (envelope, freq)             when `ret_analytic` is False
        (envelope, freq, analytic)   when `ret_analytic` is True

            envelope (Tensor) :
                       The envelope function, with its shape same as `x`.
            freq     (Tensor) :
                       The instantaneous freqency function measured in Hz, with its shape
                       same as `x`.
            analytic (Tensor) :
                       The analytic (complex) signal, with its shape same as `x`.
    '''


    analytic = hilbert(x)
    envelope = analytic.abs()
    en_raw_abs = envelope - torch.mean(envelope)
    es_raw = torch.fft.fft(en_raw_abs)
    es_raw_abs = torch.abs(es_raw) * 2 / len(en_raw_abs)
    sub = torch.cat((analytic[..., 1:] - analytic[..., :-1],
                     (analytic[..., -1] - analytic[..., -2]).unsqueeze(-1)
                     ), axis=-1)
    add = torch.cat((analytic[..., 1:] + analytic[..., :-1],
                     2 * analytic[..., -1].unsqueeze(-1)
                     ), axis=-1)
    freq = 2 * fs * ((sub / add).imag)
    freq[freq.isinf()] = 0
    del sub, add
    freq /= (2 * math.pi)
    return (es_raw_abs, freq) if not ret_analytic else (envelope, freq, analytic)



class CLASSBD(nn.Module):
    def __init__(self, l_input=2048, fs=50000, pulse_config=None, use_pag=True) -> object:
        super(CLASSBD, self).__init__()
        self.fs = fs
        self.use_pag = use_pag
        self.qtfilter = nn.Sequential(
            nn.AvgPool1d(1, 1),
            ConvQuadraticOperation(1, 16, 127, 1, 'same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            ConvQuadraticOperation(16, 1, 127, 1, 'same'),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(1, 1),
            nn.Sigmoid()
        )
        
        # 使用脉冲注意力配置（如果提供）
        if self.use_pag:
            if pulse_config is None:
                # 默认配置
                self.pulse_attn_gate = PulseAttentionGate(
                    kernel_sizes=[15, 31, 63],  # 多尺度核大小，适应不同周期的故障信号
                    threshold=3.0,              # 初始阈值，会在训练中自适应调整
                    epsilon=1e-6
                )
            else:
                # 使用传入的配置
                kernel_sizes = pulse_config.get('kernel_sizes', [15, 31, 63])
                threshold = pulse_config.get('threshold', 3.0)
                
                # 处理是否使用多尺度
                if not pulse_config.get('multiscale', True):
                    # 如果不使用多尺度，则只用中间尺度（通常是31）
                    if len(kernel_sizes) >= 2:
                        kernel_sizes = [kernel_sizes[1]]  # 使用中间尺度
                    else:
                        kernel_sizes = [31]  # 默认尺度
                        
                self.pulse_attn_gate = PulseAttentionGate(
                    kernel_sizes=kernel_sizes,
                    threshold=threshold,
                    epsilon=1e-6
                )
        else:
            # 关闭PAG时创建一个占位属性，便于下游代码做hasattr检查
            self.pulse_attn_gate = None
            
        self.filter1 = nn.Linear(l_input, l_input)

    def funcKurtosis(self, y, halfFilterlength=32):
        # 确保y具有正确的维度 [batch, channels, sequence]
        if y.dim() == 1:  # 如果是一维张量 [sequence]
            y = y.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, sequence]
        elif y.dim() == 2:  # 如果是二维张量 [batch or channel, sequence]
            if y.size(0) == 1:  # 如果第一维是batch
                y = y.unsqueeze(1)  # 变为 [batch, 1, sequence]
            else:  # 如果第一维是channel
                y = y.unsqueeze(0)  # 变为 [1, channel, sequence]
        
        # 现在y应该有3个维度 [batch, channels, sequence]
        batch_size, channels, seq_len = y.shape
        
        # 确保序列长度足够进行切片操作
        if seq_len <= 2 * halfFilterlength:
            # 序列太短，改为使用全序列
            y_2 = y - torch.mean(y, dim=-1, keepdim=True)
            num = seq_len
        else:
            # 正常进行切片操作
            y_1 = y[:, :, halfFilterlength:-halfFilterlength]
            y_2 = y_1 - torch.mean(y_1, dim=-1, keepdim=True)
            num = y_1.size(-1)
        
        # 计算峭度
        y_num = torch.sum(torch.pow(y_2, 4), dim=-1) / num
        std = torch.sqrt(torch.sum(torch.pow(y_2, 2), dim=-1) / num)
        
        # 避免除以零
        epsilon = 1e-10
        std = std + epsilon
        
        y_dem = torch.pow(std, 4)
        loss = y_num / y_dem
        
        return nn.Parameter(loss.mean())


    def g_lplq(self, y, p=2, q=4):
        es_raw_abs, _ = get_envelope_frequency(y, fs=self.fs)
        p = torch.tensor(p)
        q = torch.tensor(q)
        obj = torch.sign(torch.log(q / p)) * (torch.norm(es_raw_abs, p, dim=-1) / torch.norm(es_raw_abs, q, dim=-1)) ** p
        return nn.Parameter(obj.mean())

    def forward(self, x):
        # time filter
        a1 = self.qtfilter(x)
        
        if self.use_pag:
            # 应用改进的多尺度脉冲注意力门控
            # 现在可以同时关注不同周期的故障特征
            a1_enhanced, pag_mask = self.pulse_attn_gate(a1, return_mask=True)
            # 可以访问注意力权重的组成部分，用于监控训练过程
            kurtosis_attention = self.pulse_attn_gate.last_kurtosis_attention
            cycle_weight = self.pulse_attn_gate.last_cycle_weight
            multiscale_kurtosis = self.pulse_attn_gate.last_multiscale_kurtosis
        else:
            # 关闭PAG时直接透传并使用全1掩码
            a1_enhanced = a1
            pag_mask = torch.ones_like(a1_enhanced)
            kurtosis_attention = None
            cycle_weight = None
            multiscale_kurtosis = None
        
        # 使用增强后的信号计算峭度损失
        k = self.funcKurtosis(a1_enhanced)
        
        # frequency filter
        f1 = torch.fft.fft2(a1_enhanced - torch.mean(a1_enhanced, dim=-1).unsqueeze(1))
        f1 = abs(f1)
        es = self.filter1(f1)
        es_raw_abs, _ = get_envelope_frequency(es, self.fs)
        a2 = abs(torch.fft.ifft2(es))
        g = self.g_lplq(es_raw_abs)

        # 返回增强后的结果和中间特征用于可视化分析
        return a2, -k, g, a1, pag_mask

 

   




















