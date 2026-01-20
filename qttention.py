"""
导出 QCNN 的数据驱动注意力（QAttention）

**Direct Activation** 定义：
  A_Q(t)   = mean_c |F_raw^c(t)|
  A_cal(t) = mean_c |F_cal^c(t)|,  F_cal = F_raw ⊙ M_phy

输出：
- data/qmaps/{dataset}_snr_{snr}/time_filter_maps_class_{j}.csv
  每个样本占两行：[raw_qattn, calibrated_qattn]
- data/qmaps/{dataset}_snr_{snr}/predict.csv, truelabel.csv
- data/input/{dataset}_snr_{snr}/input_class_{j}.csv（每类取测试集样本均值，用于三轨图）
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from Model.BDCNN import BDWDCNN
from utils.JNUProcessing import JNU_Processing
from utils.PUProcessing import Paderborn_Processing


def random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _ensure_1d_length(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    if x.shape[0] == length:
        return x
    out = np.zeros(length, dtype=x.dtype)
    copy_len = min(length, x.shape[0])
    out[:copy_len] = x[:copy_len]
    return out


def main() -> None:
    random_seed(42)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    config = {
        "dataset": "Paderborn",   # "JNU" or "Paderborn"
        "add_noise": "Gaussian",  # Gaussian, pink, Laplace, airplane, truck
        "snr": 0,
        "class_num": 14,
        "batch_size": 96,
        "chosen_dataset": "N09_M07_F10",  # only for Paderborn
    }

    if config["dataset"] == "JNU":
        config["class_num"] = 10
    else:
        config["class_num"] = 14

    pulse_config = {
        "multiscale": True,
        "kernel_sizes": [15, 31, 63],
        "threshold": 3.0,
    }

    model = BDWDCNN(config["class_num"], pulse_config=pulse_config).to(device)
    run_path = f'Models/{config["dataset"]}_best_checkpoint_bdcnn.pth'
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"未找到模型权重：{run_path}。请先训练并保存 checkpoint。")

    state = torch.load(run_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if config["dataset"] == "JNU":
        _, _, Test_X, _, _, Test_Y = JNU_Processing(
            file_path=os.path.join("data", config["dataset"]),
            noise=config["add_noise"],
            snr=config["snr"],
        )
    else:
        _, _, Test_X, _, _, Test_Y = Paderborn_Processing(
            file_path=os.path.join("data", config["dataset"]),
            load=config["chosen_dataset"],
            noise=config["add_noise"],
            snr=config["snr"],
        )

    # tensor -> numpy for later grouping
    length = int(Test_X.shape[-1])
    y_true = Test_Y.detach().cpu().numpy()

    # 推理并收集每个样本的 raw/cal QAttention
    raw_qattn_list: list[np.ndarray] = []
    cal_qattn_list: list[np.ndarray] = []
    y_pred_list: list[int] = []

    with torch.no_grad():
        for start in range(0, len(Test_X), config["batch_size"]):
            end = min(start + config["batch_size"], len(Test_X))
            batch_x = Test_X[start:end].float().to(device)
            outputs = model(batch_x)  # (y_hat, k, g, feat_qcnn, pag_mask)
            y_hat, feat_qcnn, pag_mask = outputs[0], outputs[3], outputs[4]

            y_pred = torch.argmax(y_hat, dim=1).detach().cpu().numpy().tolist()
            y_pred_list.extend(y_pred)

            # Direct Activation
            raw_qattn = torch.abs(feat_qcnn).mean(dim=1)  # [B, L]
            cal_feat = feat_qcnn * pag_mask
            cal_qattn = torch.abs(cal_feat).mean(dim=1)   # [B, L]

            raw_np = raw_qattn.detach().cpu().numpy()
            cal_np = cal_qattn.detach().cpu().numpy()

            for i in range(raw_np.shape[0]):
                raw_qattn_list.append(_ensure_1d_length(raw_np[i], length))
                cal_qattn_list.append(_ensure_1d_length(cal_np[i], length))

    # 组织按类别保存：每个样本两行 [raw, calibrated]
    result_dir = f'data/qmaps/{config["dataset"]}_snr_{config["snr"]}'
    os.makedirs(result_dir, exist_ok=True)

    for cls in range(config["class_num"]):
        idx = np.where(y_true == cls)[0]
        rows = []
        for i in idx:
            rows.append(raw_qattn_list[i])
            rows.append(cal_qattn_list[i])
        arr = np.vstack(rows) if rows else np.zeros((0, length))
        pd.DataFrame(arr).to_csv(
            f"{result_dir}/time_filter_maps_class_{cls}.csv",
            header=False,
            index=False,
        )

    pd.DataFrame(y_pred_list).to_csv(f"{result_dir}/predict.csv", header=False, index=False)
    pd.DataFrame(y_true).to_csv(f"{result_dir}/truelabel.csv", header=False, index=False)

    # 保存每类输入代表波形：取测试集该类所有样本均值
    input_dir = f'data/input/{config["dataset"]}_snr_{config["snr"]}'
    os.makedirs(input_dir, exist_ok=True)
    x_np = Test_X.detach().cpu().numpy()
    if x_np.ndim == 3:
        x_np = x_np.squeeze(1)
    for cls in range(config["class_num"]):
        idx = np.where(y_true == cls)[0]
        if len(idx) == 0:
            rep = np.zeros(length)
        else:
            rep = np.mean(x_np[idx], axis=0).reshape(-1)
        pd.DataFrame(_ensure_1d_length(rep, length)).to_csv(
            f"{input_dir}/input_class_{cls}.csv",
            header=False,
            index=False,
        )

    print(f"[OK] QAttention 已导出到: {result_dir}/")


if __name__ == "__main__":
    main()

def attention_compose(attention_maps, output_length, stride=1, efficient_stride=None):
    """
    组合多个窗口位置的注意力图，生成完整的序列注意力
    
    Args:
        attention_maps: 列表，包含每个窗口位置的注意力响应，每个元素形状为[batch_size, out_channels]
        output_length: 输出序列的长度
        stride: 窗口滑动的步长（用于输出映射）
        efficient_stride: 实际计算时使用的步长（用于窗口位置计算）
        
    Returns:
        组合后的注意力图，形状为[batch_size, out_channels, output_length]
    """
    if not attention_maps:
        print("警告: 没有注意力图可组合")
        return None
    
    # 使用实际的计算步长来确定窗口位置
    actual_stride = efficient_stride if efficient_stride is not None else stride
    
    # 获取批次大小和输出通道数
    first_map = attention_maps[0]
    
    # 安全检查 - 确保我们可以获取形状
    if not isinstance(first_map, torch.Tensor):
        print(f"警告: 注意力图不是张量，而是 {type(first_map)}")
        # 尝试转换为张量
        try:
            first_map = torch.tensor(first_map)
        except:
            print("无法转换为张量，使用零张量")
            return torch.zeros(1, 1, output_length)
    
    # 处理不同维度的输入
    if first_map.dim() == 1:  # [features]
        batch_size = 1
        out_channels = 1
        first_map = first_map.unsqueeze(0)  # 添加batch维度
    elif first_map.dim() == 2:  # [batch, features] 或 [batch, channels]
        batch_size, features = first_map.shape
        out_channels = features  # 假设features是通道数
    elif first_map.dim() == 3:  # [batch, channels, 1]
        batch_size, out_channels, _ = first_map.shape
        first_map = first_map.squeeze(-1)
    else:
        print(f"警告: 注意力图维度异常: {first_map.shape}")
        batch_size = first_map.shape[0]
        out_channels = 1
    
    # 创建输出张量 [batch_size, out_channels, output_length]
    combined_map = torch.zeros(batch_size, out_channels, output_length, device=first_map.device)
    
    # 构建贡献计数张量，用于计算重叠区域的平均值
    contribution_count = torch.zeros(output_length, device=first_map.device)
    
    print(f"组合注意力图: 输出形状=[{batch_size}, {out_channels}, {output_length}], 窗口数={len(attention_maps)}, 实际步长={actual_stride}")
    
    # 对每个窗口的注意力响应分配到其对应的序列位置
    for i, attention in enumerate(attention_maps):
        # 使用实际计算步长来确定窗口中心位置
        center_pos = i * actual_stride
        
        # 确保attention的形状正确
        if not isinstance(attention, torch.Tensor):
            try:
                attention = torch.tensor(attention, device=first_map.device)
            except:
                print(f"警告: 跳过窗口 {i}，无法转换为张量")
                continue
                
        # 标准化维度
        if attention.dim() == 1:  # [features]
            attention = attention.unsqueeze(0)  # [1, features]
        elif attention.dim() == 3 and attention.shape[2] == 1:  # [batch, channels, 1]
            attention = attention.squeeze(2)  # [batch, channels]
            
        # 确保attention形状与期望相匹配
        if attention.shape[0] != batch_size:
            print(f"警告: 窗口 {i} 批次大小不匹配: {attention.shape[0]} vs {batch_size}")
            # 尝试广播或复制
            if attention.shape[0] == 1:
                attention = attention.repeat(batch_size, 1)
        
        if attention.shape[1] != out_channels:
            print(f"警告: 窗口 {i} 通道数不匹配: {attention.shape[1]} vs {out_channels}")
            # 尝试调整
            if attention.shape[1] > out_channels:
                attention = attention[:, :out_channels]
            else:
                temp = torch.zeros(batch_size, out_channels, device=attention.device)
                temp[:, :attention.shape[1]] = attention
                attention = temp
                
        # 将注意力值分布到对应的序列位置范围内
        # 使用输出步长来确定在最终序列中的分布范围
        start_pos = max(0, center_pos - stride // 2)
        end_pos = min(output_length, center_pos + stride // 2 + 1)
        
        # 将注意力值分布到这个范围内的所有位置
        for pos in range(start_pos, end_pos):
            combined_map[:, :, pos] += attention
            contribution_count[pos] += 1
    
    # 处理重叠区域：计算平均值
    valid_positions = contribution_count > 0
    for b in range(batch_size):
        for c in range(out_channels):
            combined_map[b, c, valid_positions] /= contribution_count[valid_positions]
    
    # 对于没有贡献的位置，使用插值填充
    empty_positions = contribution_count == 0
    if empty_positions.any():
        print(f"使用插值填充 {empty_positions.sum().item()} 个空白位置")
        for b in range(batch_size):
            for c in range(out_channels):
                # 对每个通道进行一维插值
                valid_indices = torch.where(valid_positions)[0]
                if len(valid_indices) > 1:
                    # 使用线性插值填充空白位置
                    valid_values = combined_map[b, c, valid_indices]
                    # 创建插值函数
                    all_indices = torch.arange(output_length, device=first_map.device, dtype=torch.float)
                    #interpolated = torch.interp(all_indices, valid_indices.float(), valid_values)
                    all_indices_np = all_indices.cpu().numpy()
                    valid_indices_np = valid_indices.float().cpu().numpy()
                    valid_values_np = valid_values.cpu().numpy()
                    interpolated_np = np.interp(all_indices_np, valid_indices_np, valid_values_np)
                    interpolated = torch.from_numpy(interpolated_np).to(valid_values.device)
                    combined_map[b, c, :] = interpolated
    
    return combined_map

def attention_map(x, wr, wg, wb, br=None, bg=None, stride=1, padding='same', kernel_size=None, efficient_stride=None):
    """
    计算二次卷积的注意力图，实现二次神经元的物理可解释性
    
    Args:
        x: 输入张量 [batch, in_channels, seq_len]
        wr: 线性项权重r [out_channels, in_channels, kernel_size]
        wg: 线性项权重g [out_channels, in_channels, kernel_size]
        wb: 二次项权重b [out_channels, in_channels, kernel_size]
        br: 线性偏置r (可选) [out_channels]
        bg: 线性偏置g (可选) [out_channels]
        stride: 步长 (保留原始默认值1以兼容现有代码)
        padding: 填充类型，'same'表示保持输出大小与输入相同
        kernel_size: 卷积核大小（如果为None，则从权重自动获取）
        efficient_stride: 内部计算使用的步长，用于提高效率而不改变API (可选)
        
    Returns:
        注意力图 [batch, out_channels, seq_len]
    """
    # 安全检查 - 确保输入是张量
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"输入必须是张量，而不是 {type(x)}")
        
    # 安全检查 - 确保权重是张量
    if not all(isinstance(w, torch.Tensor) for w in [wr, wg, wb]):
        raise ValueError("所有权重必须是张量")
    
    # 获取基本参数
    batch_size, in_channels, seq_len = x.shape
    out_channels, w_in_channels, w_kernel_size = wr.shape
    
    # 验证权重形状
    if wb.shape != wr.shape or wg.shape != wr.shape:
        raise ValueError(f"权重形状不一致: wr={wr.shape}, wg={wg.shape}, wb={wb.shape}")
    
    # 动态获取kernel_size
    if kernel_size is None:
        kernel_size = w_kernel_size
    elif w_kernel_size != kernel_size:
        print(f"警告: 权重核大小({w_kernel_size})与参数指定的核大小({kernel_size})不一致，使用权重的实际核大小")
        kernel_size = w_kernel_size
    
    # 使用efficient_stride提高计算效率
    compute_stride = efficient_stride if efficient_stride is not None else stride
    
    # 计算padding大小以保持输出维度
    if padding == 'same':
        padding_size = kernel_size // 2
    elif isinstance(padding, int):
        padding_size = padding
    else:
        padding_size = 0
    
    # 打印输入和参数信息
    print(f"二次卷积注意力计算: 输入=[{batch_size}, {in_channels}, {seq_len}], "
          f"权重=[{out_channels}, {w_in_channels}, {kernel_size}], "
          f"步长={stride}, 计算步长={compute_stride}, 填充={padding_size}")
    
    # 调整输入通道与权重匹配
    if in_channels != w_in_channels:
        print(f"输入通道({in_channels})与权重通道({w_in_channels})不匹配，进行调整...")
        
        # 根据情况进行调整
        if in_channels == 1 and w_in_channels > 1:
            # 单通道输入与多通道权重：复制输入通道
            print(f"复制单通道输入以匹配多通道权重")
            x = x.repeat(1, w_in_channels, 1)
            in_channels = w_in_channels
        elif in_channels > 1 and w_in_channels == 1:
            # 多通道输入与单通道权重：复制权重通道
            print(f"复制单通道权重以匹配多通道输入")
            wr = wr.repeat(1, in_channels, 1)
            wg = wg.repeat(1, in_channels, 1)
            wb = wb.repeat(1, in_channels, 1)
            w_in_channels = in_channels
        else:
            # 其他情况：尝试最大程度匹配
            print(f"通道数不匹配，执行自适应调整")
            if in_channels > w_in_channels:
                # 截取输入通道
                x = x[:, :w_in_channels, :]
                in_channels = w_in_channels
            else:
                # 截取权重通道
                wr = wr[:, :in_channels, :]
                wg = wg[:, :in_channels, :]
                wb = wb[:, :in_channels, :]
                w_in_channels = in_channels
    
    # 添加填充
    if padding_size > 0:
        padding_tensor = torch.zeros(batch_size, in_channels, padding_size, device=x.device)
        x_padded = torch.cat([padding_tensor, x, padding_tensor], dim=2)
    else:
        x_padded = x
    
    # 计算有效序列长度和输出序列长度
    padded_seq_len = x_padded.shape[2]
    output_length = seq_len  # 对于'same'填充，输出长度等于输入长度
    
    # 计算卷积窗口数，使用计算步长来减少窗口总数
    n_windows = (padded_seq_len - kernel_size) // compute_stride + 1
    print(f"计算窗口总数: {n_windows}，内部步长={compute_stride}，每个窗口大小={kernel_size}")
    
    # 存储每个位置的注意力响应
    attention_responses = []
    
    # 对每个窗口位置计算注意力响应
    for i in range(n_windows):
        window_start = i * compute_stride
        window_end = window_start + kernel_size
        
        # 确保窗口不超出序列范围
        if window_end <= padded_seq_len:
            # 提取当前窗口
            window = x_padded[:, :, window_start:window_end]  # [batch, in_channels, kernel_size]
            
            # 计算二次项
            window_squared = torch.pow(window, 2)  # [batch, in_channels, kernel_size]
            
            try:
                # 使用PyTorch的卷积操作计算二次神经元的三个分量
                # 计算二次项贡献
                quad_term = F.conv1d(
                    window_squared, 
                    wb, 
                    bias=None, 
                    stride=1, 
                    padding=0, 
                    groups=1
                )  # [batch, out_channels, 1]
                
                # 计算两个线性项贡献
                linear_r = F.conv1d(
                    window, 
                    wr, 
                    bias=br, 
                    stride=1, 
                    padding=0, 
                    groups=1
                )  # [batch, out_channels, 1]
                
                linear_g = F.conv1d(
                    window, 
                    wg, 
                    bias=bg, 
                    stride=1, 
                    padding=0, 
                    groups=1
                )  # [batch, out_channels, 1]
                
                # 组合三个分量计算最终二次神经元响应
                linear_term = linear_r * linear_g  # 乘性线性项
                response = quad_term + linear_term  # 总响应
                
                # 压缩多余的维度
                response = response.squeeze(-1)  # [batch, out_channels]
                
                # 记录当前窗口的注意力响应
                attention_responses.append(response)
                
                # 打印部分窗口的结果作为示例
                if i == 0 or i == n_windows // 2 or i == n_windows - 1:
                    print(f"窗口 #{i}/{n_windows-1}: response形状={response.shape}")
                
            except RuntimeError as e:
                print(f"PyTorch卷积计算失败，尝试手动实现: {e}")
                
                # 手动实现二次神经元计算
                batch_size, in_channels, kernel_size = window.shape
                out_channels = wr.shape[0]
                
                # 对每个样本和输出通道手动计算卷积
                responses_manual = torch.zeros(batch_size, out_channels, device=x.device)
                
                for b in range(batch_size):
                    for o in range(out_channels):
                        # 计算二次项
                        quad_response = 0.0
                        for c in range(in_channels):
                            # 窗口的平方与相应权重的内积
                            quad_response += torch.sum(window_squared[b, c] * wb[o, c])
                        
                        # 计算线性项
                        linear_r_response = 0.0
                        linear_g_response = 0.0
                        for c in range(in_channels):
                            # 窗口与相应权重的内积
                            linear_r_response += torch.sum(window[b, c] * wr[o, c])
                            linear_g_response += torch.sum(window[b, c] * wg[o, c])
                        
                        # 添加偏置（如果有）
                        if br is not None:
                            linear_r_response += br[o]
                        if bg is not None:
                            linear_g_response += bg[o]
                        
                        # 计算总响应
                        responses_manual[b, o] = quad_response + (linear_r_response * linear_g_response)
                
                # 添加到响应列表
                attention_responses.append(responses_manual)
                
                if i == 0 or i == n_windows // 2 or i == n_windows - 1:
                    print(f"窗口 #{i}/{n_windows-1} (手动计算): response形状={responses_manual.shape}")
    
    # 组合所有窗口位置的注意力响应，生成完整序列的注意力图
    if attention_responses:
        print(f"组合{len(attention_responses)}个窗口的注意力响应...")
        # 传递efficient_stride参数给attention_compose函数
        attention_map = attention_compose(attention_responses, output_length, stride, efficient_stride)
        
        return attention_map
    else:
        print("错误: 未生成任何注意力响应")
        return torch.zeros(batch_size, out_channels, output_length, device=x.device)

def visualize_separate_analysis(Test_X, time_filter_map, input_data, num_classes, result_dir, layer1_filter_map=None):
    """
    生成分离的可视化图，每种图类型单独保存：
    1. 原始信号波形图
    2. 第一层QCNN注意力图（1→16通道的输出）
    3. 第一层QCNN处理后的信号波形图
    4. 第二层QCNN注意力图（16→1通道的输出，即feat_qcnn）
    5. 校准后QCNN注意力图（feat_qcnn * pag_mask）
    
    Args:
        Test_X: 测试数据
        time_filter_map: 第二层QCNN和校准后注意力图字典 {class_idx: [n_samples*2, length]}
        input_data: 输入信号字典 {class_idx: [length]}
        num_classes: 类别数
        result_dir: 结果保存目录
        layer1_filter_map: 第一层QCNN注意力图字典 {class_idx: [n_samples, length]}，可选
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    for class_idx in range(num_classes):
        if class_idx not in input_data or len(input_data[class_idx]) == 0:
            continue
            
        # 1. 原始信号波形图
        plt.figure(figsize=(12, 6))
        plt.plot(input_data[class_idx], 'b-', linewidth=1.5)
        plt.title(f'Class {class_idx} - Original Signal Waveform', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{result_dir}/class_{class_idx}_original_signal.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 第一层QCNN注意力图（1→16通道）
        plt.figure(figsize=(12, 6))
        if layer1_filter_map is not None and class_idx in layer1_filter_map and len(layer1_filter_map[class_idx]) > 0:
            # 使用第一个样本
            layer1_attn = layer1_filter_map[class_idx][0] if layer1_filter_map[class_idx].ndim > 1 else layer1_filter_map[class_idx]
            plt.plot(layer1_attn, 'r-', linewidth=1.5, label='Layer 1 QCNN (1→16 channels)')
            plt.title(f'Class {class_idx} - Layer 1 QCNN Attention Map (1→16 channels)', fontsize=14)
        else:
            plt.plot(np.zeros(len(input_data[class_idx])), 'r-', linewidth=1.5)
            plt.title(f'Class {class_idx} - Layer 1 QCNN Attention Map (No Data)', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Attention Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{result_dir}/class_{class_idx}_layer1_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 第一层QCNN处理后的信号波形图
        plt.figure(figsize=(12, 6))
        if layer1_filter_map is not None and class_idx in layer1_filter_map and len(layer1_filter_map[class_idx]) > 0:
            signal_len = len(input_data[class_idx])
            layer1_attn = layer1_filter_map[class_idx][0] if layer1_filter_map[class_idx].ndim > 1 else layer1_filter_map[class_idx]
            attention_len = len(layer1_attn)
            
            if signal_len == attention_len:
                processed_signal = input_data[class_idx] * layer1_attn
            else:
                if attention_len > signal_len:
                    attention_weights = layer1_attn[:signal_len]
                else:
                    attention_weights = np.resize(layer1_attn, signal_len)
                processed_signal = input_data[class_idx] * attention_weights
        else:
            processed_signal = input_data[class_idx]
            
        plt.plot(processed_signal, 'g-', linewidth=1.5)
        plt.title(f'Class {class_idx} - Signal After Layer 1 QCNN Processing', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{result_dir}/class_{class_idx}_after_layer1.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 第二层QCNN注意力图（16→1通道，即feat_qcnn）
        plt.figure(figsize=(12, 6))
        if class_idx in time_filter_map and len(time_filter_map[class_idx]) > 0:
            # time_filter_map格式：[Layer2_Raw_sample0, Calibrated_sample0, Layer2_Raw_sample1, ...]
            # 取偶数索引行（Layer 2 Raw）
            layer2_attn = time_filter_map[class_idx][0, :]  # 第一个样本的第二层Raw注意力
            plt.plot(layer2_attn, 'm-', linewidth=1.5, label='Layer 2 QCNN (16→1 channel, Raw)')
            plt.title(f'Class {class_idx} - Layer 2 QCNN Attention Map (16→1 channel, Raw)', fontsize=14)
        else:
            plt.plot(np.zeros(len(input_data[class_idx])), 'm-', linewidth=1.5)
            plt.title(f'Class {class_idx} - Layer 2 QCNN Attention Map (No Data)', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Attention Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{result_dir}/class_{class_idx}_layer2_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 校准后QCNN注意力图（新增）
        plt.figure(figsize=(12, 6))
        if class_idx in time_filter_map and len(time_filter_map[class_idx]) > 1:
            # 取奇数索引行（Calibrated）
            calibrated_attn = time_filter_map[class_idx][1, :]  # 第一个样本的校准后注意力
            plt.plot(calibrated_attn, 'c-', linewidth=1.5, label='Calibrated QCNN (PAG-calibrated)')
            plt.title(f'Class {class_idx} - Calibrated QCNN Attention Map (PAG-calibrated)', fontsize=14)
        else:
            plt.plot(np.zeros(len(input_data[class_idx])), 'c-', linewidth=1.5)
            plt.title(f'Class {class_idx} - Calibrated QCNN Attention Map (No Data)', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Attention Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{result_dir}/class_{class_idx}_calibrated_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"分离的可视化图已保存到 {result_dir}/")
    print(f"每个类别生成5张独立的图片：")
    print(f"- class_X_original_signal.png: 原始信号波形图")
    print(f"- class_X_layer1_attention.png: 第一层QCNN注意力图（1→16通道）")
    print(f"- class_X_after_layer1.png: 第一层QCNN处理后的信号波形图")
    print(f"- class_X_layer2_attention.png: 第二层QCNN注意力图（16→1通道，Raw）")
    print(f"- class_X_calibrated_attention.png: 校准后QCNN注意力图（PAG校准）")

def main():
    random_seed(42)
    model_name = 'bdcnn'
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # 设置数据集和模型参数
    config = {
        'dataset': 'Paderborn',      # JNU or Paderborn
        'add_noise': 'Gaussian',
        'snr': 0,             # 信噪比
        'class_num': 14,      # JNU默认是10类，Paderborn是14类
        'batch_size': 96
    }
    
    if config['dataset'] == 'Paderborn':
        config['class_num'] = 14
    
    # 初始化模型（需要与训练时使用相同的pulse_config）
    # 注意：这里使用默认配置，如果训练时使用了自定义pulse_config，需要保持一致
    pulse_config = {
        'multiscale': True,
        'kernel_sizes': [15, 31, 63],
        'threshold': 3.0
    }
    model = BDWDCNN(config['class_num'], pulse_config=pulse_config)
    run_path = f'Models/{config["dataset"]}_best_checkpoint_{model_name}.pth'  # 模型路径
    
    print(f"加载模型: {run_path}")
    # 加载模型权重
    try:
        best_model_dict = torch.load(run_path, map_location=device)
        model.load_state_dict(best_model_dict)
        model.to(device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 数据准备
    print(f"加载{config['dataset']}数据集，噪声类型: {config['add_noise']}，信噪比: {config['snr']} dB")
    if config['dataset'] == 'JNU':
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = JNU_Processing(
            file_path=os.path.join('data', config['dataset']),
            noise=config['add_noise'], 
            snr=config['snr']
        )
        length = 2048  # 根据JNU数据的长度设置
    elif config['dataset'] == 'Paderborn':  # 修改这里，使用elif而不是else
        chosen_dataset = 'N09_M07_F10'  # 修改为您需要的子数据集
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = Paderborn_Processing(
            file_path=os.path.join('data', config['dataset']),
            load=chosen_dataset,
            noise=config['add_noise'], 
            snr=config['snr']
        )
        length = 2048  # 根据Paderborn数据的长度设置
    else:
        raise ValueError(f"不支持的数据集: {config['dataset']}") 
    # 转换为张量并移动到设备
    Test_X = Test_X.to(device)
    Test_Y = Test_Y.to(device)
    
    print(f"测试数据形状: {Test_X.shape}, 标签形状: {Test_Y.shape}")
    
    # 创建数据加载器
    test_dataset = CustomTensorDataset(Test_X, Test_Y)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    
    # ========== 提取ConvQuadraticOperation层的权重用于物理可解释性分析 ==========
    print("提取二次卷积层的权重用于物理可解释性分析...")
    qtfilter = model.classbd.qtfilter
    qcnn_layers = []
    for i, layer in enumerate(qtfilter):
        if isinstance(layer, ConvQuadraticOperation):
            qcnn_layers.append({
                'layer_idx': i,
                'layer': layer,
                'weight_r': layer.weight_r,
                'weight_g': layer.weight_g,
                'weight_b': layer.weight_b,
                'bias_r': layer.bias_r if layer.bias else None,
                'bias_g': layer.bias_g if layer.bias else None,
                'stride': layer.stride,
                'padding': layer.padding,
                'kernel_size': layer.kernel_size
            })
            print(f"找到ConvQuadraticOperation层 {i}: kernel_size={layer.kernel_size}, "
                  f"in_channels={layer.in_features}, out_channels={layer.out_features}")
    
    # 检查是否找到ConvQuadraticOperation层
    if len(qcnn_layers) == 0:
        raise ValueError("错误: 未找到ConvQuadraticOperation层！模型结构可能不正确。")
    
    if len(qcnn_layers) < 2:
        raise ValueError(f"错误: 期望找到2个ConvQuadraticOperation层，但只找到{len(qcnn_layers)}个！")
    
    # 第一层QCNN：1→16通道（索引0）
    # 第二层QCNN：16→1通道（索引1）
    qcnn_layer1 = qcnn_layers[0]  # 第一层QCNN
    qcnn_layer2 = qcnn_layers[1]  # 第二层QCNN
    print(f"第一层QCNN: {qcnn_layer1['layer'].in_features}→{qcnn_layer1['layer'].out_features}通道")
    print(f"第二层QCNN: {qcnn_layer2['layer'].in_features}→{qcnn_layer2['layer'].out_features}通道")
    
    # 模型推理和注意力图提取
    # 实现"Calibration & Validation"逻辑：
    # 
    # 时域滤波器qtfilter包含两层ConvQuadraticOperation：
    #   - QCNN Layer 1: 1→16通道（第一层QCNN）
    #   - QCNN Layer 2: 16→1通道（第二层QCNN，输出feat_qcnn）
    #
    # 我们提取三层注意力：
    #   - Layer 1 Attention: 第一层QCNN的注意力（1→16通道的输出）
    #   - Layer 2 Attention: 第二层QCNN的注意力（16→1通道的输出，即feat_qcnn）
    #   - Calibrated Attention: 经过PAG校准的注意力（feat_qcnn * pag_mask）
    #
    print("开始提取QCNN注意力图（第一层、第二层和校准后三层）...")
    y_pre = []
    qcnn_layer1_attention_maps = []      # 存储第一层QCNN注意力图
    qcnn_layer2_attention_maps = []      # 存储第二层QCNN注意力图（Raw，未校准）
    qcnn_attention_maps_calibrated = []  # 存储校准后QCNN注意力图（经过PAG）
    
    with torch.no_grad():
        # 获取数据的基本信息
        length = Test_X.shape[-1]
        y = Test_Y.cpu()
        
        # 分批处理以避免内存溢出
        batch_size = config['batch_size']
        
        for start_idx in range(0, len(Test_X), batch_size):
            end_idx = min(start_idx + batch_size, len(Test_X))
            print(f"处理批次 {start_idx//batch_size + 1}/{(len(Test_X)-1)//batch_size + 1}: 样本 {start_idx}-{end_idx-1}")
            
            batch_X = Test_X[start_idx:end_idx]
            
            # 前向传播，获取模型输出
            outputs = model(batch_X)
            # outputs = (y_hat, k, g, feat_qcnn, pag_mask)
            predictions = torch.argmax(outputs[0], dim=1)
            y_pre.extend(predictions.cpu().numpy())
            
            # 提取feat_qcnn (即a1，qtfilter的输出) 和 pag_mask (PAG注意力权重)
            feat_qcnn = outputs[3]  # [B, C, L] - 原始QCNN特征
            pag_mask = outputs[4]    # [B, C, L] - PAG注意力权重
            
            # 调试信息：检查feat_qcnn和pag_mask的统计信息
            if start_idx == 0 and end_idx == batch_size:
                print(f"feat_qcnn形状: {feat_qcnn.shape}")
                print(f"feat_qcnn统计: min={feat_qcnn.min().item():.6f}, max={feat_qcnn.max().item():.6f}, mean={feat_qcnn.mean().item():.6f}, std={feat_qcnn.std().item():.6f}")
                print(f"pag_mask形状: {pag_mask.shape}")
                print(f"pag_mask统计: min={pag_mask.min().item():.6f}, max={pag_mask.max().item():.6f}, mean={pag_mask.mean().item():.6f}, std={pag_mask.std().item():.6f}")
                
                # ========== PAG内部状态诊断 ==========
                pag_module = model.classbd.pulse_attn_gate
                print("\n" + "="*60)
                print("PAG内部状态诊断:")
                print("="*60)
                
                # 检查阈值参数
                if hasattr(pag_module, 'threshold'):
                    threshold_value = pag_module.threshold.item()
                    print(f"PAG threshold参数值: {threshold_value:.6f}")
                
                # 检查内部状态
                if hasattr(pag_module, 'last_kurtosis'):
                    local_kurtosis = pag_module.last_kurtosis  # [B, C, L]
                    local_kurtosis_avg = local_kurtosis.mean(dim=1)  # [B, L]
                    print(f"\nLocal Kurtosis统计:")
                    print(f"  形状: {local_kurtosis.shape}")
                    print(f"  最小值: {local_kurtosis_avg.min().item():.6f}")
                    print(f"  最大值: {local_kurtosis_avg.max().item():.6f}")
                    print(f"  均值: {local_kurtosis_avg.mean().item():.6f}")
                    print(f"  标准差: {local_kurtosis_avg.std().item():.6f}")
                    
                    # 分析sigmoid输入
                    if hasattr(pag_module, 'threshold'):
                        sigmoid_input = local_kurtosis_avg - threshold_value
                        print(f"\nSigmoid输入 (local_kurtosis - threshold) 统计:")
                        print(f"  最小值: {sigmoid_input.min().item():.6f}")
                        print(f"  最大值: {sigmoid_input.max().item():.6f}")
                        print(f"  均值: {sigmoid_input.mean().item():.6f}")
                        print(f"  标准差: {sigmoid_input.std().item():.6f}")
                
                if hasattr(pag_module, 'last_kurtosis_attention'):
                    kurtosis_attn = pag_module.last_kurtosis_attention  # [B, C, L]
                    kurtosis_attn_avg = kurtosis_attn.mean(dim=1)  # [B, L]
                    print(f"\nKurtosis Attention统计:")
                    print(f"  形状: {kurtosis_attn.shape}")
                    print(f"  最小值: {kurtosis_attn_avg.min().item():.6f}")
                    print(f"  最大值: {kurtosis_attn_avg.max().item():.6f}")
                    print(f"  均值: {kurtosis_attn_avg.mean().item():.6f}")
                    print(f"  标准差: {kurtosis_attn_avg.std().item():.6f}")
                
                if hasattr(pag_module, 'last_cycle_weight') and pag_module.last_cycle_weight is not None:
                    cycle_weight = pag_module.last_cycle_weight  # [B, C, L]
                    cycle_weight_avg = cycle_weight.mean(dim=1)  # [B, L]
                    print(f"\nCycle Weight统计:")
                    print(f"  形状: {cycle_weight.shape}")
                    print(f"  最小值: {cycle_weight_avg.min().item():.6f}")
                    print(f"  最大值: {cycle_weight_avg.max().item():.6f}")
                    print(f"  均值: {cycle_weight_avg.mean().item():.6f}")
                    print(f"  标准差: {cycle_weight_avg.std().item():.6f}")
                
                # 诊断建议
                print("\n" + "="*60)
                print("诊断分析:")
                print("="*60)
                if hasattr(pag_module, 'last_kurtosis') and hasattr(pag_module, 'threshold'):
                    local_kurtosis_avg = pag_module.last_kurtosis.mean(dim=1)
                    threshold_value = pag_module.threshold.item()
                    sigmoid_input = local_kurtosis_avg - threshold_value
                    
                    if abs(sigmoid_input.mean().item()) < 0.5:
                        print("⚠️  问题1: Local Kurtosis - Threshold 的均值接近0")
                        print("   这意味着大部分区域的峭度值接近阈值，导致sigmoid输出集中在0.5附近")
                        print(f"   建议: 调整threshold参数（当前={threshold_value:.6f}），使其远离local kurtosis的均值")
                    
                    if sigmoid_input.std().item() < 1.0:
                        print("⚠️  问题2: Local Kurtosis - Threshold 的标准差太小")
                        print("   这意味着峭度值变化不够大，导致注意力权重缺乏对比度")
                        print("   建议: 检查局部峭度计算是否正确，或者信号本身缺乏脉冲特征")
                
                if hasattr(pag_module, 'last_kurtosis_attention'):
                    kurtosis_attn_avg = pag_module.last_kurtosis_attention.mean(dim=1)
                    if kurtosis_attn_avg.std().item() < 0.1:
                        print("⚠️  问题3: Kurtosis Attention的标准差太小")
                        print("   这意味着注意力权重过于均匀，无法突出重要区域")
                        print("   建议: 增加峭度计算的对比度，或者使用更强的非线性变换")
                
                if pag_mask.std().item() < 0.02:
                    print("⚠️  问题4: PAG Mask的标准差太小")
                    print("   这意味着最终注意力权重过于均匀")
                    print("   建议: 检查cycle_weight是否也过于均匀，或者调整融合方式")
                
                print("="*60 + "\n")
            
            # ========== Layer 1: 第一层QCNN的注意力 (1→16通道) ==========
            # 使用物理可解释性方法计算第一层QCNN的注意力
            # 第一层QCNN的输入是原始信号（经过AvgPool1d）
            x_input_layer1 = batch_X  # [B, 1, L]
            # 如果第一层之前有其他层（如AvgPool1d），需要先处理
            if qcnn_layer1['layer_idx'] > 0:
                for i in range(qcnn_layer1['layer_idx']):
                    x_input_layer1 = qtfilter[i](x_input_layer1)
            
            # 计算第一层QCNN的物理可解释性注意力
            qcnn_layer1_attn_physical = attention_map(
                x=x_input_layer1,
                wr=qcnn_layer1['weight_r'],
                wg=qcnn_layer1['weight_g'],
                wb=qcnn_layer1['weight_b'],
                br=qcnn_layer1['bias_r'],
                bg=qcnn_layer1['bias_g'],
                stride=qcnn_layer1['stride'],
                padding=qcnn_layer1['padding'],
                kernel_size=qcnn_layer1['kernel_size'],
                efficient_stride=1  # 对每个位置都进行计算，不跳过
            )  # [B, 16, L] - 16个通道的输出
            
            # 跨通道平均得到第一层注意力图
            qcnn_layer1_attn = qcnn_layer1_attn_physical.mean(dim=1)  # [B, L]
            
            # ========== Layer 2: 第二层QCNN的注意力 (16→1通道，即feat_qcnn) ==========
            # 使用物理可解释性方法计算第二层QCNN的注意力
            # 第二层QCNN的输入是第一层QCNN的输出（经过BN和ReLU）
            # 需要前向传播到第二层之前
            x_input_layer2 = batch_X  # [B, 1, L]
            for i in range(qcnn_layer2['layer_idx']):
                x_input_layer2 = qtfilter[i](x_input_layer2)
            # 此时x_input_layer2应该是[B, 16, L]，即第一层QCNN的输出
            
            # 计算第二层QCNN的物理可解释性注意力
            qcnn_layer2_attn_physical = attention_map(
                x=x_input_layer2,
                wr=qcnn_layer2['weight_r'],
                wg=qcnn_layer2['weight_g'],
                wb=qcnn_layer2['weight_b'],
                br=qcnn_layer2['bias_r'],
                bg=qcnn_layer2['bias_g'],
                stride=qcnn_layer2['stride'],
                padding=qcnn_layer2['padding'],
                kernel_size=qcnn_layer2['kernel_size'],
                efficient_stride=1  # 对每个位置都进行计算，不跳过
            )  # [B, 1, L] - 1个通道的输出
            
            # 跨通道平均得到第二层注意力图（实际上只有1个通道，mean只是为了统一格式）
            qcnn_layer2_attn = qcnn_layer2_attn_physical.mean(dim=1)  # [B, L]
            
            if start_idx == 0 and end_idx == batch_size:
                print(f"第一层QCNN注意力形状: {qcnn_layer1_attn_physical.shape} -> {qcnn_layer1_attn.shape}")
                print(f"第二层QCNN注意力形状: {qcnn_layer2_attn_physical.shape} -> {qcnn_layer2_attn.shape}")
            
            # ========== Calibrated Attention: 经过PAG校准的注意力 ==========
            # 这是摘要中 "calibrate and validate" 的数学体现！
            # feat_calibrated = feat_qcnn * pag_mask (逐元素相乘)
            # 物理意义：PAG通过物理约束（峭度、周期性）验证并校准第二层QCNN的注意力
            feat_calibrated = feat_qcnn * pag_mask  # [B, C, L]
            # 计算校准后的注意力图
            qcnn_attn_calibrated = torch.abs(feat_calibrated).mean(dim=1)  # [B, L]
            
            # ========== 关键诊断：检查attention_map输出和feat_qcnn的差异 ==========
            if start_idx == 0 and end_idx == batch_size:
                print("\n" + "="*60)
                print("关键诊断：attention_map输出 vs feat_qcnn")
                print("="*60)
                
                # 检查attention_map的输出（用于可视化）
                qcnn_layer2_attn_sample = qcnn_layer2_attn[0].cpu().numpy()  # 第一个样本
                print(f"\nLayer 2 QCNN Attention Map (attention_map输出) 统计:")
                print(f"  形状: {qcnn_layer2_attn_sample.shape}")
                print(f"  最小值: {qcnn_layer2_attn_sample.min():.6f}")
                print(f"  最大值: {qcnn_layer2_attn_sample.max():.6f}")
                print(f"  均值: {qcnn_layer2_attn_sample.mean():.6f}")
                print(f"  标准差: {qcnn_layer2_attn_sample.std():.6f}")
                
                # 检查feat_qcnn的实际值（用于校准）
                feat_qcnn_sample = feat_qcnn[0].squeeze().cpu().numpy()  # 第一个样本，[L]
                print(f"\nfeat_qcnn (实际QCNN输出，经过BN+ReLU+Sigmoid) 统计:")
                print(f"  形状: {feat_qcnn_sample.shape}")
                print(f"  最小值: {feat_qcnn_sample.min():.6f}")
                print(f"  最大值: {feat_qcnn_sample.max():.6f}")
                print(f"  均值: {feat_qcnn_sample.mean():.6f}")
                print(f"  标准差: {feat_qcnn_sample.std():.6f}")
                
                # 检查关键区域（950-1000时间步）
                region_start, region_end = 950, 1000
                print(f"\n关键区域 ({region_start}-{region_end}时间步) 对比:")
                print(f"  attention_map输出:")
                print(f"    均值: {qcnn_layer2_attn_sample[region_start:region_end].mean():.6f}")
                print(f"    最大值: {qcnn_layer2_attn_sample[region_start:region_end].max():.6f}")
                print(f"    最小值: {qcnn_layer2_attn_sample[region_start:region_end].min():.6f}")
                print(f"  feat_qcnn:")
                print(f"    均值: {feat_qcnn_sample[region_start:region_end].mean():.6f}")
                print(f"    最大值: {feat_qcnn_sample[region_start:region_end].max():.6f}")
                print(f"    最小值: {feat_qcnn_sample[region_start:region_end].min():.6f}")
                print(f"  pag_mask:")
                pag_mask_sample = pag_mask[0].squeeze().cpu().numpy()
                print(f"    均值: {pag_mask_sample[region_start:region_end].mean():.6f}")
                print(f"    最小值: {pag_mask_sample[region_start:region_end].min():.6f}")
                print(f"    最大值: {pag_mask_sample[region_start:region_end].max():.6f}")
                print(f"  校准后注意力 (feat_qcnn * pag_mask):")
                calibrated_sample = qcnn_attn_calibrated[0].cpu().numpy()
                print(f"    均值: {calibrated_sample[region_start:region_end].mean():.6f}")
                print(f"    最大值: {calibrated_sample[region_start:region_end].max():.6f}")
                print(f"    最小值: {calibrated_sample[region_start:region_end].min():.6f}")
                
                print("\n" + "="*60)
                print("问题分析:")
                print("="*60)
                print("⚠️  发现：attention_map输出和feat_qcnn不是同一个量级！")
                print("  原因：")
                print("    1. attention_map计算的是原始二次神经元响应（未归一化，可能为负）")
                print("    2. feat_qcnn经过了BN+ReLU+Sigmoid，值被压缩到[0,1]")
                print("    3. 因此attention_map的输出可能很大（几百）或为负，而feat_qcnn只有0.5-0.9")
                print("  影响：")
                print("    - Layer 2 QCNN注意力图显示峰值800是正常的（原始响应）")
                print("    - 校准后注意力值范围0-0.8也是正常的（feat_qcnn * pag_mask）")
                print("    - 两者可视化不一致，但这是预期的行为")
                print("="*60 + "\n")
            
            # 调试信息：检查注意力图的统计信息
            if start_idx == 0 and end_idx == batch_size:
                print(f"第一层QCNN注意力形状: {qcnn_layer1_attn.shape}")
                print(f"第一层QCNN注意力统计: min={qcnn_layer1_attn.min().item():.6f}, max={qcnn_layer1_attn.max().item():.6f}, mean={qcnn_layer1_attn.mean().item():.6f}, std={qcnn_layer1_attn.std().item():.6f}")
                print(f"第二层QCNN注意力形状: {qcnn_layer2_attn.shape}")
                print(f"第二层QCNN注意力统计: min={qcnn_layer2_attn.min().item():.6f}, max={qcnn_layer2_attn.max().item():.6f}, mean={qcnn_layer2_attn.mean().item():.6f}, std={qcnn_layer2_attn.std().item():.6f}")
                print(f"校准后注意力形状: {qcnn_attn_calibrated.shape}")
                print(f"校准后注意力统计: min={qcnn_attn_calibrated.min().item():.6f}, max={qcnn_attn_calibrated.max().item():.6f}, mean={qcnn_attn_calibrated.mean().item():.6f}, std={qcnn_attn_calibrated.std().item():.6f}")
            
            # 转换为numpy并保存
            qcnn_layer1_attn_np = qcnn_layer1_attn.cpu().numpy()  # [B, L]
            qcnn_layer2_attn_np = qcnn_layer2_attn.cpu().numpy()  # [B, L]
            qcnn_attn_calibrated_np = qcnn_attn_calibrated.cpu().numpy()  # [B, L]
            
            # 确保长度正确并保存
            for i in range(qcnn_layer2_attn_np.shape[0]):
                # 第一层注意力
                attn_map_layer1 = qcnn_layer1_attn_np[i]  # [L]
                # 第二层注意力（Raw，未校准）
                attn_map_layer2 = qcnn_layer2_attn_np[i]  # [L]
                # 校准后注意力
                attn_map_calibrated = qcnn_attn_calibrated_np[i]  # [L]
                
                # 调试信息：检查单个样本的统计信息
                if start_idx == 0 and i == 0:
                    print(f"样本0 第一层QCNN注意力统计: min={attn_map_layer1.min():.6f}, max={attn_map_layer1.max():.6f}, mean={attn_map_layer1.mean():.6f}, std={attn_map_layer1.std():.6f}")
                    print(f"样本0 第二层QCNN注意力统计: min={attn_map_layer2.min():.6f}, max={attn_map_layer2.max():.6f}, mean={attn_map_layer2.mean():.6f}, std={attn_map_layer2.std():.6f}")
                    print(f"样本0 校准后注意力统计: min={attn_map_calibrated.min():.6f}, max={attn_map_calibrated.max():.6f}, mean={attn_map_calibrated.mean():.6f}, std={attn_map_calibrated.std():.6f}")
                
                # 确保长度为length
                def ensure_length(attn_map, length):
                    if len(attn_map) != length:
                        temp = np.zeros(length)
                        copy_len = min(len(attn_map), length)
                        temp[:copy_len] = attn_map[:copy_len]
                        return temp
                    return attn_map
                
                attn_map_layer1 = ensure_length(attn_map_layer1, length)
                attn_map_layer2 = ensure_length(attn_map_layer2, length)
                attn_map_calibrated = ensure_length(attn_map_calibrated, length)
                
                # 保存三层注意力
                qcnn_layer1_attention_maps.append(attn_map_layer1)
                qcnn_layer2_attention_maps.append(attn_map_layer2)
                qcnn_attention_maps_calibrated.append(attn_map_calibrated)
            
            # 清理内存
            torch.cuda.empty_cache() if use_gpu else None
    
    print(f"成功提取 {len(qcnn_layer1_attention_maps)} 个第一层QCNN注意力图")
    print(f"成功提取 {len(qcnn_layer2_attention_maps)} 个第二层QCNN注意力图")
    print(f"成功提取 {len(qcnn_attention_maps_calibrated)} 个校准后QCNN注意力图")
    
    # 保存时域滤波器注意力图
    time_filter_map = {}
    num_classes = config['class_num']
    
    print(f"收集到的第一层QCNN样本总数: {len(qcnn_layer1_attention_maps)}")
    print(f"收集到的第二层QCNN样本总数: {len(qcnn_layer2_attention_maps)}")
    print(f"收集到的校准后样本总数: {len(qcnn_attention_maps_calibrated)}")
    print(f"测试集样本总数: {len(y)}")
    
    # 确保所有列表的长度与测试集样本数一致
    def align_attention_maps(attention_maps, name, target_length, length):
        if len(attention_maps) != target_length:
            print(f"警告: {name}注意力图数量({len(attention_maps)})与测试集样本数({target_length})不一致")
            if len(attention_maps) < target_length:
                missing_count = target_length - len(attention_maps)
                print(f"缺少 {missing_count} 个样本，使用零填充")
                for _ in range(missing_count):
                    attention_maps.append(np.zeros(length))
            else:
                print(f"多余 {len(attention_maps) - target_length} 个样本，截取前{target_length}个")
                attention_maps = attention_maps[:target_length]
        return attention_maps
    
    qcnn_layer1_attention_maps = align_attention_maps(qcnn_layer1_attention_maps, "第一层QCNN", len(y), length)
    qcnn_layer2_attention_maps = align_attention_maps(qcnn_layer2_attention_maps, "第二层QCNN", len(y), length)
    qcnn_attention_maps_calibrated = align_attention_maps(qcnn_attention_maps_calibrated, "校准后QCNN", len(y), length)
    
    # 遍历每个类别
    for j in range(num_classes):
        # 找出属于该类别的样本索引
        idx = np.where(y.numpy() == j)[0]
        print(f"类别 {j}: 包含 {len(idx)} 个样本")
        
        # 创建该类别样本的处理结果列表
        processed_maps = []
        
        # 处理该类别的每个样本
        for sample_idx in idx:
            # 获取样本的三层注意力图
            attn_map_layer1 = qcnn_layer1_attention_maps[sample_idx]  # [L] - 第一层QCNN注意力
            attn_map_layer2 = qcnn_layer2_attention_maps[sample_idx]  # [L] - 第二层QCNN注意力（Raw）
            attn_map_calibrated = qcnn_attention_maps_calibrated[sample_idx]  # [L] - 校准后注意力
            
            # 确保是一维数组
            if attn_map_layer1.ndim > 1:
                attn_map_layer1 = attn_map_layer1.flatten()
            if attn_map_layer2.ndim > 1:
                attn_map_layer2 = attn_map_layer2.flatten()
            if attn_map_calibrated.ndim > 1:
                attn_map_calibrated = attn_map_calibrated.flatten()
            
            # 确保长度为length
            def ensure_length(attn_map, length):
                if len(attn_map) != length:
                    temp = np.zeros(length)
                    copy_len = min(len(attn_map), length)
                    temp[:copy_len] = attn_map[:copy_len]
                    return temp
                return attn_map
            
            attn_map_layer1 = ensure_length(attn_map_layer1, length)
            attn_map_layer2 = ensure_length(attn_map_layer2, length)
            attn_map_calibrated = ensure_length(attn_map_calibrated, length)
            
            # 保存两层数据（用于与triple_track_visualization.py兼容）：
            # Layer 1 (Raw): 第二层QCNN注意力（16→1通道的输出，即feat_qcnn，未校准）
            # Layer 2 (Calibrated): 校准后注意力（feat_qcnn * pag_mask，经过PAG校准）
            # 注意：第一层QCNN注意力单独保存到layer1_qcnn_attention_class_*.csv
            map_data = np.stack([attn_map_layer2, attn_map_calibrated])  # [2, L]
            
            # 添加到处理结果列表
            processed_maps.append(map_data)
        
        # 将所有样本堆叠成一个大数组，形状为 [n_samples*2, length]
        # 每个样本贡献2行：[Layer2_Raw, Calibrated]
        if len(processed_maps) > 0:
            stacked_maps = np.vstack(processed_maps)
            print(f"  类别 {j}: 成功堆叠 {len(processed_maps)} 个样本，形状: {stacked_maps.shape}")
            time_filter_map[j] = stacked_maps
        else:
            print(f"  警告: 类别 {j} 没有任何样本")
            time_filter_map[j] = np.zeros((0, length))

    # 保存为CSV之前，打印每个类别的行数
    print("\n保存为CSV之前，每个类别的行数:")
    total_rows = 0
    for j in range(num_classes):
        rows = time_filter_map[j].shape[0]
        samples = rows // 2  # 每个样本占2行
        print(f"类别 {j}: {rows} 行 ({samples} 个样本)")
        total_rows += rows
    
    print(f"所有类别总行数: {total_rows} (总样本数: {total_rows//2})")
    print(f"原始测试集样本总数: {len(y)}")
    print(f"期望的总行数: {len(y)*2}")
    
    # 检查是否有缺失的样本
    if total_rows < len(y)*2:
        print(f"警告: 总行数 {total_rows} 小于期望的 {len(y)*2}，可能有样本丢失")
    
    # 保存为CSV
    result_dir = f'data/qmaps/{config["dataset"]}_snr_{config["snr"]}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存第一层QCNN注意力（如果可用）
    if len(qcnn_layer1_attention_maps) > 0:
        layer1_filter_map = {}
        for j in range(num_classes):
            idx = np.where(y.numpy() == j)[0]
            if len(idx) > 0:
                layer1_samples = [qcnn_layer1_attention_maps[i] for i in idx if i < len(qcnn_layer1_attention_maps)]
                if layer1_samples:
                    layer1_filter_map[j] = np.stack(layer1_samples)
                else:
                    layer1_filter_map[j] = np.zeros((0, length))
            else:
                layer1_filter_map[j] = np.zeros((0, length))
        
        for j in range(num_classes):
            pd.DataFrame(layer1_filter_map[j]).to_csv(
                f'{result_dir}/layer1_qcnn_attention_class_{j}.csv',
                header=False,
                index=False
            )
        print(f"第一层QCNN注意力图已保存到 {result_dir}/layer1_qcnn_attention_class_*.csv")
    
    for j in range(num_classes):
        # 保存时域滤波器注意力图（第二层Raw和Calibrated）
        # 格式：每两行代表一个样本 [Layer2_Raw, Layer2_Calibrated]
        pd.DataFrame(time_filter_map[j]).to_csv(
            f'{result_dir}/time_filter_maps_class_{j}.csv', 
            header=False,
            index=False
        )
    
    pd.DataFrame(y_pre).to_csv(f'{result_dir}/predict.csv', header=False)
    pd.DataFrame(Test_Y.cpu().numpy()).to_csv(f'{result_dir}/truelabel.csv', header=False)
    
    # 保存输入信号
    input_data = {}
    for j in range(num_classes):
        idx = np.where(Test_Y.cpu().numpy() == j)[0]
        if len(idx) > 0:
            # 确保维度正确 - 获取样本，保持原始形状，再展平
            samples = Test_X[idx].cpu().numpy()
            # 只保存每个类的第一个样本作为代表
            # 修改为保存每个类所有样本的平均值
            class_avg = np.mean(samples, axis=0).squeeze()
            input_data[j] = class_avg.reshape(-1)  # 展平为一维数组
            print(f"类别 {j}: 保存了 {len(idx)} 个样本的平均值，形状 {input_data[j].shape}")
        else:
            print(f"类别 {j}: 没有样本，使用零填充")
            input_data[j] = np.zeros(length)
    
    input_dir = f'data/input/{config["dataset"]}_snr_{config["snr"]}'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    for j in range(num_classes):
        pd.DataFrame(input_data[j]).to_csv(
            f'{input_dir}/input_class_{j}.csv', 
            header=False
        )
    
    # 可视化注意力图
    #visualize_attention_maps(time_filter_map, num_classes, result_dir)
    
    # 生成分离的可视化图
    # 准备第一层QCNN注意力图
    layer1_filter_map = {}
    for j in range(num_classes):
        idx = np.where(Test_Y.cpu().numpy() == j)[0]
        if len(idx) > 0:
            layer1_samples = [qcnn_layer1_attention_maps[i] for i in idx]
            layer1_filter_map[j] = np.stack(layer1_samples)
        else:
            layer1_filter_map[j] = np.zeros((0, length))
    
    visualize_separate_analysis(Test_X, time_filter_map, input_data, num_classes, result_dir, layer1_filter_map)
    
    # ========== 定量验证：计算注意力有效性指标 ==========
    print("\n" + "="*60)
    print("开始定量验证注意力有效性...")
    print("="*60)
    
    # 计算注意力峰值与信号能量的相关性
    attention_validation_results = {}
    for class_idx in range(num_classes):
        if class_idx not in time_filter_map or time_filter_map[class_idx].shape[0] == 0:
            continue
        
        # 获取该类别的样本索引
        idx = np.where(Test_Y.cpu().numpy() == class_idx)[0]
        if len(idx) == 0:
            continue
        
        # 获取该类别的注意力图（使用Calibrated版本，即第二层）
        if time_filter_map[class_idx].shape[0] >= 2:
            # 提取Calibrated QCNN注意力（偶数索引行，即Layer 2）
            calibrated_attn = time_filter_map[class_idx][1::2]  # 取所有偶数索引行
        else:
            calibrated_attn = time_filter_map[class_idx]
        
        # 获取对应的原始信号
        class_signals = Test_X[idx].cpu().numpy().squeeze()  # [N, L] 或 [N, 1, L]
        if class_signals.ndim == 3:
            class_signals = class_signals.squeeze(1)  # [N, L]
        
        # 确保注意力图数量与信号数量匹配
        min_samples = min(len(calibrated_attn), len(class_signals))
        calibrated_attn = calibrated_attn[:min_samples]
        class_signals = class_signals[:min_samples]
        
        # 计算指标
        correlations = []
        peak_alignments = []
        attention_sparsities = []
        
        for i in range(min_samples):
            attn = calibrated_attn[i]
            signal = class_signals[i]
            
            # 1. 计算注意力与信号能量的相关性
            signal_energy = signal ** 2
            # 归一化
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            energy_norm = (signal_energy - signal_energy.min()) / (signal_energy.max() - signal_energy.min() + 1e-8)
            # 计算皮尔逊相关系数
            if len(attn_norm) == len(energy_norm):
                corr = np.corrcoef(attn_norm, energy_norm)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            # 2. 计算注意力峰值与信号峰值的位置对齐度
            # 找到注意力峰值位置（前5%的最大值）
            top_k = max(1, int(len(attn) * 0.05))
            attn_peaks = np.argsort(attn)[-top_k:]
            # 找到信号能量峰值位置
            signal_peaks = np.argsort(signal_energy)[-top_k:]
            # 计算重叠度（有多少峰值位置在彼此的邻域内）
            alignment = 0
            for ap in attn_peaks:
                # 检查是否有信号峰值在注意力峰值的邻域内（±50个采样点）
                neighborhood = np.abs(signal_peaks - ap) <= 50
                if np.any(neighborhood):
                    alignment += 1
            peak_alignments.append(alignment / top_k)
            
            # 3. 计算注意力稀疏性（Gini系数）
            attn_sorted = np.sort(attn)
            n = len(attn_sorted)
            cumsum = np.cumsum(attn_sorted)
            if cumsum[-1] > 0:
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                attention_sparsities.append(gini)
        
        # 计算统计量
        if correlations:
            attention_validation_results[class_idx] = {
                'correlation_mean': np.mean(correlations),
                'correlation_std': np.std(correlations),
                'peak_alignment_mean': np.mean(peak_alignments),
                'peak_alignment_std': np.std(peak_alignments),
                'sparsity_mean': np.mean(attention_sparsities),
                'sparsity_std': np.std(attention_sparsities),
                'num_samples': min_samples
            }
    
    # 保存验证结果
    validation_report_path = f'{result_dir}/attention_validation_report.txt'
    with open(validation_report_path, 'w', encoding='utf-8') as f:
        f.write("QCNN注意力定量验证报告\n")
        f.write("="*60 + "\n\n")
        f.write("说明：\n")
        f.write("1. Correlation: 注意力与信号能量的皮尔逊相关系数（越高越好，表示注意力关注高能量区域）\n")
        f.write("2. Peak Alignment: 注意力峰值与信号峰值的位置对齐度（0-1，越高越好）\n")
        f.write("3. Sparsity (Gini): 注意力的稀疏性（0-1，越高表示注意力越集中在少数位置）\n\n")
        f.write("-"*60 + "\n\n")
        
        for class_idx, metrics in attention_validation_results.items():
            f.write(f"类别 {class_idx} (样本数: {metrics['num_samples']}):\n")
            f.write(f"  注意力-能量相关性: {metrics['correlation_mean']:.4f} ± {metrics['correlation_std']:.4f}\n")
            f.write(f"  峰值对齐度: {metrics['peak_alignment_mean']:.4f} ± {metrics['peak_alignment_std']:.4f}\n")
            f.write(f"  注意力稀疏性: {metrics['sparsity_mean']:.4f} ± {metrics['sparsity_std']:.4f}\n\n")
        
        # 计算总体统计
        if attention_validation_results:
            all_corrs = [m['correlation_mean'] for m in attention_validation_results.values()]
            all_alignments = [m['peak_alignment_mean'] for m in attention_validation_results.values()]
            all_sparsities = [m['sparsity_mean'] for m in attention_validation_results.values()]
            
            f.write("总体统计:\n")
            f.write(f"  平均相关性: {np.mean(all_corrs):.4f}\n")
            f.write(f"  平均峰值对齐度: {np.mean(all_alignments):.4f}\n")
            f.write(f"  平均稀疏性: {np.mean(all_sparsities):.4f}\n")
    
    print(f"定量验证报告已保存到: {validation_report_path}")
    for class_idx, metrics in attention_validation_results.items():
        print(f"类别 {class_idx}: 相关性={metrics['correlation_mean']:.3f}, "
              f"对齐度={metrics['peak_alignment_mean']:.3f}, "
              f"稀疏性={metrics['sparsity_mean']:.3f}")
    
    print("="*60)
    
    print(f"时域滤波器二次卷积的注意力图已保存到 {result_dir}/")
    print(f"预测标签已保存到 {result_dir}/predict.csv")
    print(f"真实标签已保存到 {result_dir}/truelabel.csv")
    print(f"输入信号已保存到 {input_dir}/")

if __name__ == '__main__':
    main()