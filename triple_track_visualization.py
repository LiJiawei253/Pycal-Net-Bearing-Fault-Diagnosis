"""
三轨并行可视化脚本
用于可视化PAG对QCNN的校准过程

校准过程可视化框架：
1. Raw QCNN (数据驱动的提案):
   - 使用二次卷积操作捕获信号的二次特征
   - 对高能量区域（包括噪声）都有较高响应
   - 需要物理验证来过滤噪声

2. PAG (物理驱动的过滤器):
   - 基于物理先验的注意力机制
   - 使用局部峭度（kurtosis）检测冲击特征
   - 使用周期性约束（cycle weight）验证故障的周期性
   - 只在物理上有效的区域产生高响应

3. Calibrated QCNN (校准后):
   - Calibrated = Raw QCNN × PAG（特征层面校准）
   - 物理意义：PAG过滤噪声，保留故障冲击
   - 结果：噪声被抑制，故障特征被增强

可视化结构（三轨展示校准过程）：
- Track 1（上）：Raw Signal（原始含噪声信号）
- Track 2（中）：Before Calibration（校准前对比）
  - Raw QCNN: 数据驱动的提案（包含噪声响应）
  - PAG: 物理驱动的过滤器（仅响应物理有效区域）
- Track 3（下）：After Calibration（校准后）
  - Calibrated QCNN: 噪声被抑制，故障冲击被保留

使用方法：
1. 确保已经运行过 qttention.py 和 pttention.py 生成注意力数据
2. 修改脚本顶部的配置参数（DATASET, SNR等）
3. 运行: python triple_track_visualization.py
4. 结果保存在 visualizations/ 目录下

输出说明：
- 每个类别生成前5个样本的可视化图
- 图片格式：PNG，300 DPI，适合论文使用
- 文件名格式：triple_track_class_{class_idx}_sample_{sample_idx}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import torch
from utils.PUProcessing import Paderborn_Processing
from utils.JNUProcessing import JNU_Processing

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
DATASET = 'Paderborn'  # 'Paderborn' 或 'JNU'
SNR = 0               # 信噪比
NUM_CLASSES = 14       # 类别数（Paderborn=14, JNU=10）
CHOSEN_DATASET = 'N09_M07_F10'  # Paderborn子数据集
NOISE_TYPE = 'Gaussian'  # 噪声类型

# 数据路径
PT_DIR = f'data/ptmaps/{DATASET}_snr_{SNR}'
QT_DIR = f'data/qmaps/{DATASET}_snr_{SNR}'
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_signal(dataset, snr, noise_type, chosen_dataset=None):
    """加载原始测试信号"""
    print(f"加载原始信号数据...")
    if dataset == 'Paderborn':
        _, _, Test_X, _, _, Test_Y = Paderborn_Processing(
            file_path=os.path.join('data', dataset),
            load=chosen_dataset,
            noise=noise_type,
            snr=snr
        )
    elif dataset == 'JNU':
        _, _, Test_X, _, _, Test_Y = JNU_Processing(
            file_path=os.path.join('data', dataset),
            noise=noise_type,
            snr=snr
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # 转换为numpy数组
    if isinstance(Test_X, torch.Tensor):
        Test_X = Test_X.cpu().numpy()
    if isinstance(Test_Y, torch.Tensor):
        Test_Y = Test_Y.cpu().numpy()
    
    # 移除通道维度 [N, 1, L] -> [N, L]
    if Test_X.ndim == 3:
        Test_X = Test_X.squeeze(1)
    
    print(f"原始信号形状: {Test_X.shape}, 标签形状: {Test_Y.shape}")
    return Test_X, Test_Y

def load_attention_data(class_idx, pt_dir, qt_dir):
    """加载QCNN和PAG注意力数据"""
    # 加载PAG注意力数据
    pt_path = f'{pt_dir}/attention_weights_class_{class_idx}.csv'
    try:
        pt_data_raw = pd.read_csv(pt_path, header=None).values
        # 检查并跳过索引列
        if pt_data_raw.shape[1] > 2048:
            first_col = pt_data_raw[:, 0]
            if np.allclose(first_col, np.arange(len(first_col)), atol=1e-6):
                pt_data = pt_data_raw[:, 1:]
            else:
                pt_data = pt_data_raw
        else:
            pt_data = pt_data_raw
    except FileNotFoundError:
        print(f"警告: {pt_path} 未找到")
        return None, None
    
    # 加载QCNN注意力数据
    qt_path = f'{qt_dir}/time_filter_maps_class_{class_idx}.csv'
    try:
        qt_data_raw = pd.read_csv(qt_path, header=None).values
        # 检查并跳过索引列
        if qt_data_raw.shape[1] > 2048:
            first_col = qt_data_raw[:, 0]
            if np.allclose(first_col[:min(10, len(first_col))], 
                          np.arange(min(10, len(first_col))), atol=1e-6):
                qt_data = qt_data_raw[:, 1:]
            else:
                qt_data = qt_data_raw
        else:
            qt_data = qt_data_raw
        
        # QCNN数据有两层（Layer 1和Layer 2），根据新的Calibration逻辑：
        # Layer 1: Raw QCNN Attention (未校准，展示噪声敏感性)
        # Layer 2: Calibrated QCNN Attention (经过PAG校准，展示物理一致性)
        # 数据格式：[Layer1_sample0, Layer2_sample0, Layer1_sample1, Layer2_sample1, ...]
        # 根据qttention.py的保存逻辑，每个样本占2行，交替排列
        num_samples = len(pt_data)
        if qt_data.shape[0] == 2 * num_samples:
            # 如果是两层交替排列，取偶数索引行（Layer 1: Raw QCNN）
            qt_data_raw = qt_data[::2]  # 取每两行的第一行（Layer 1: Raw）
            qt_data_calibrated = qt_data[1::2]  # 取每两行的第二行（Layer 2: Calibrated）
            # 返回Raw和Calibrated两层数据
            return pt_data, (qt_data_raw, qt_data_calibrated)
        elif qt_data.shape[0] == num_samples:
            # 如果只有一层，直接使用（向后兼容）
            qt_data = qt_data
            return pt_data, qt_data
        else:
            # 如果格式不同，尝试取前num_samples行
            print(f"  警告: QCNN数据行数({qt_data.shape[0]})与PAG数据行数({num_samples})不匹配")
            if qt_data.shape[0] > num_samples:
                qt_data = qt_data[:num_samples]
                return pt_data, qt_data
            else:
                # 如果QCNN数据更少，可能需要填充或跳过
                print(f"  跳过类别 {class_idx}（QCNN数据不足）")
                return None, None
    except FileNotFoundError:
        print(f"警告: {qt_path} 未找到")
        return None, None
    
    return pt_data, qt_data

def normalize_attention(attn):
    """归一化注意力权重（逐样本min-max归一化）"""
    attn = np.asarray(attn)
    if attn.size == 0:
        return attn
    
    if attn.ndim == 1:
        attn_min = attn.min()
        attn_max = attn.max()
        if attn_max - attn_min < 1e-12:
            return np.ones_like(attn) * 0.5 if abs(attn_max) > 1e-12 else np.zeros_like(attn)
        normalized = (attn - attn_min) / (attn_max - attn_min + 1e-8)
        return normalized
    else:
        normalized = np.zeros_like(attn)
        for i in range(attn.shape[0]):
            normalized[i] = normalize_attention(attn[i])
        return normalized

def plot_triple_track(raw_signal, raw_qcnn_attn, pag_attn, calibrated_qcnn_attn, 
                      sample_idx, class_idx, output_dir, fs=50000):
    """
    绘制三轨并行可视化图：展示PAG对QCNN的校准过程
    
    Args:
        raw_signal: 原始信号 [L]
        raw_qcnn_attn: Raw QCNN注意力（未校准）[L]
        pag_attn: PAG注意力（物理过滤器）[L]
        calibrated_qcnn_attn: Calibrated QCNN注意力（校准后）[L]
        sample_idx: 样本索引
        class_idx: 类别索引
        output_dir: 输出目录
        fs: 采样频率
    """
    # 确保长度一致
    min_len = min(len(raw_signal), len(raw_qcnn_attn), len(pag_attn), len(calibrated_qcnn_attn))
    raw_signal = raw_signal[:min_len]
    raw_qcnn_attn = raw_qcnn_attn[:min_len]
    pag_attn = pag_attn[:min_len]
    calibrated_qcnn_attn = calibrated_qcnn_attn[:min_len]
    
    # 归一化注意力
    raw_qcnn_attn_norm = normalize_attention(raw_qcnn_attn)
    pag_attn_norm = normalize_attention(pag_attn)
    calibrated_qcnn_attn_norm = normalize_attention(calibrated_qcnn_attn)
    
    # 创建时间轴（毫秒）
    time_ms = np.arange(min_len) / fs * 1000
    
    # 创建图形（高分辨率，适合论文）
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.5, 1.2], hspace=0.35)
    
    # 设置全局字体大小
    title_fontsize = 16
    label_fontsize = 14
    legend_fontsize = 12
    tick_fontsize = 11
    
    # ========== 第一轨：Raw Signal ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_ms, raw_signal, 'k-', linewidth=1.2, alpha=0.8)
    ax1.set_ylabel('Amplitude', fontsize=label_fontsize, fontweight='bold')
    ax1.set_title('Track 1: Raw Signal (Noisy)', fontsize=title_fontsize, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=tick_fontsize)
    # 设置y轴范围，使信号居中
    y_range = np.abs(raw_signal).max() * 1.2
    ax1.set_ylim(-y_range, y_range)
    
    # ========== 第二轨：Dual Sources (Before Calibration) ==========
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    # Raw QCNN Attention - 数据驱动的提案，可能包含噪声响应
    # 使用蓝色，展示校准前的原始响应
    ax2.plot(time_ms, raw_qcnn_attn_norm, 'b-', linewidth=2.2, alpha=0.75, 
             label='Raw QCNN (Data-driven, with noise)', zorder=2)
    ax2.fill_between(time_ms, 0, raw_qcnn_attn_norm, alpha=0.25, color='blue', zorder=1)
    
    # PAG Attention - 物理驱动的过滤器，只在物理上有效的区域高响应
    # 使用红色，作为物理Ground Truth参考
    ax2.plot(time_ms, pag_attn_norm, 'r-', linewidth=2.8, alpha=0.95, 
             label='PAG (Physics-driven Filter)', zorder=3)
    ax2.fill_between(time_ms, 0, pag_attn_norm, alpha=0.35, color='red', zorder=2)
    
    ax2.set_ylabel('Normalized Attention', fontsize=label_fontsize, fontweight='bold')
    ax2.set_title('Track 2: Before Calibration (Raw QCNN vs PAG)', 
                  fontsize=title_fontsize, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='black', fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(labelsize=tick_fontsize)
    
    # ========== 第三轨：After Calibration ==========
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    # Calibrated QCNN Attention - 校准后，噪声被抑制，故障冲击被保留
    # 使用绿色，展示PAG校准后的清晰结果
    ax3.plot(time_ms, calibrated_qcnn_attn_norm, 'g-', linewidth=3.0, alpha=0.95, 
             label='Calibrated QCNN (Raw QCNN × PAG)', zorder=2)
    ax3.fill_between(time_ms, 0, calibrated_qcnn_attn_norm, alpha=0.45, color='green', zorder=1)
    ax3.set_xlabel('Time (ms)', fontsize=label_fontsize, fontweight='bold')
    ax3.set_ylabel('Normalized Attention', fontsize=label_fontsize, fontweight='bold')
    ax3.set_title('Track 3: After Calibration (Noise Suppressed, Fault Preserved)', 
                  fontsize=title_fontsize, fontweight='bold', pad=15)
    ax3.legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.9,
               edgecolor='black', fancybox=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(-0.05, 1.1)
    ax3.tick_params(labelsize=tick_fontsize)
    
    # 添加解释性文本
    fig.text(0.5, 0.02, 
             'Calibration Process: Raw QCNN proposes candidate regions (may include noise) → PAG filters with physics (kurtosis + periodicity) → Calibrated QCNN suppresses noise and preserves fault impacts',
             ha='center', fontsize=10, style='italic', alpha=0.7)
    
    # 保存图片
    output_path = f'{output_dir}/triple_track_class_{class_idx}_sample_{sample_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {output_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("三轨并行可视化脚本")
    print("=" * 60)
    
    # 加载原始信号
    raw_signals, labels = load_raw_signal(DATASET, SNR, NOISE_TYPE, CHOSEN_DATASET)
    
    # 为每个类别生成可视化
    for class_idx in range(NUM_CLASSES):
        print(f"\n处理类别 {class_idx}...")
        
        # 加载注意力数据
        pt_data, qt_data = load_attention_data(class_idx, PT_DIR, QT_DIR)
        if pt_data is None or qt_data is None:
            print(f"  跳过类别 {class_idx}（数据未找到）")
            continue
        
        # 处理QCNN数据格式（可能是元组 (raw, calibrated) 或单个数组）
        if isinstance(qt_data, tuple):
            qt_data_raw, qt_data_calibrated = qt_data
            print(f"  检测到两层QCNN数据: Raw和Calibrated")
        else:
            # 向后兼容：如果只有一层，使用它作为Raw
            qt_data_raw = qt_data
            qt_data_calibrated = qt_data  # 如果没有Calibrated，使用Raw作为占位符
            print(f"  检测到单层QCNN数据（向后兼容模式）")
        
        # 找到该类别的样本索引
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) == 0:
            print(f"  跳过类别 {class_idx}（无样本）")
            continue
        
        # 确保数据对齐
        num_samples = min(len(class_indices), len(pt_data), len(qt_data_raw), len(qt_data_calibrated))
        if num_samples == 0:
            print(f"  跳过类别 {class_idx}（数据为空）")
            continue
        
        print(f"  找到 {num_samples} 个样本")
        
        # 为每个样本生成可视化（可以选择前几个或全部）
        samples_to_plot = min(5, num_samples)  # 每个类别绘制前5个样本
        for i in range(samples_to_plot):
            sample_idx = class_indices[i]
            
            # 获取数据
            raw_signal = raw_signals[sample_idx]
            pag_attn = pt_data[i]
            qcnn_attn_raw = qt_data_raw[i]  # Raw QCNN（未校准）
            qcnn_attn_calibrated = qt_data_calibrated[i]  # Calibrated QCNN（校准后）
            
            # ========== 可视化校准过程 ==========
            # Track 1: Raw Signal（原始含噪声信号）
            # Track 2: Raw QCNN vs PAG（校准前对比）
            #   - Raw QCNN: 数据驱动的提案，对高能量区域（包括噪声）都响应
            #   - PAG: 物理驱动的过滤器，只对高峭度+周期性区域响应
            # Track 3: Calibrated QCNN（校准后）
            #   - Calibrated = Raw QCNN × PAG（特征层面）
            #   - 结果：噪声被抑制，故障冲击被保留
            
            # 绘制三轨图
            plot_triple_track(
                raw_signal=raw_signal,
                raw_qcnn_attn=qcnn_attn_raw,  # Raw QCNN
                pag_attn=pag_attn,  # PAG
                calibrated_qcnn_attn=qcnn_attn_calibrated,  # Calibrated QCNN
                sample_idx=sample_idx,
                class_idx=class_idx,
                output_dir=OUTPUT_DIR,
                fs=50000
            )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()

