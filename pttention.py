"""
导出 PAG 的物理注意力（PAttention）与局部峭度（可选）

本脚本只做一件事：运行一次前向传播并保存 PAG 输出的注意力掩码：
- data/ptmaps/{dataset}_snr_{snr}/attention_weights_class_{j}.csv
- data/ptmaps/{dataset}_snr_{snr}/local_kurtosis_raw_class_{j}.csv
- data/ptmaps/{dataset}_snr_{snr}/predict.csv, truelabel.csv

说明：
- PulseAttentionGate 内部缓存的 `last_attention` 为最终融合后的 M_phy（用于三轨图的 PAttention）。
- 为了与可视化一致，这里对通道维做均值： [B, C, L] -> [B, L]。
"""

import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch

from Model.BDCNN import BDWDCNN
from Model.PulseAttentionGate import PulseAttentionGate
from utils.JNUProcessing import JNU_Processing
from utils.PUProcessing import Paderborn_Processing


def random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    # 数据
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

    length = int(Test_X.shape[-1])
    y_true = Test_Y.detach().cpu().numpy()

    attn_list: List[np.ndarray] = []
    kurt_list: List[np.ndarray] = []

    def pag_hook(module, _inputs, _outputs):
        # 取最终注意力
        attn = getattr(module, "last_attention", None)
        if attn is not None:
            attn_np = attn.detach().cpu().numpy()
            if attn_np.ndim == 3:  # [B, C, L] -> [B, L]
                attn_np = attn_np.mean(axis=1)
            for i in range(attn_np.shape[0]):
                attn_list.append(_ensure_1d_length(attn_np[i], length))

        # 取局部峭度（用于可选可视化/分析）
        lk = getattr(module, "last_kurtosis", None)
        if lk is not None:
            lk_np = lk.detach().cpu().numpy()
            if lk_np.ndim == 3:  # [B, C, L] -> [B, L]
                lk_np = lk_np.mean(axis=1)
            for i in range(lk_np.shape[0]):
                kurt_list.append(_ensure_1d_length(lk_np[i], length))

    handle = None
    for m in model.modules():
        if isinstance(m, PulseAttentionGate):
            handle = m.register_forward_hook(pag_hook)
            break
    if handle is None:
        raise RuntimeError("未找到 PulseAttentionGate 模块，无法导出 PAttention。")

    y_pred_list: List[int] = []
    with torch.no_grad():
        for start in range(0, len(Test_X), config["batch_size"]):
            end = min(start + config["batch_size"], len(Test_X))
            batch_x = Test_X[start:end].float().to(device)
            outputs = model(batch_x)
            y_hat = outputs[0]
            y_pred_list.extend(torch.argmax(y_hat, dim=1).detach().cpu().numpy().tolist())

    handle.remove()

    # 对齐（异常情况下用零填充）
    n = len(y_true)
    if len(attn_list) != n:
        if len(attn_list) < n:
            attn_list.extend([np.zeros(length)] * (n - len(attn_list)))
        else:
            attn_list = attn_list[:n]
    if len(kurt_list) != n:
        if len(kurt_list) < n:
            kurt_list.extend([np.zeros(length)] * (n - len(kurt_list)))
        else:
            kurt_list = kurt_list[:n]

    result_dir = f'data/ptmaps/{config["dataset"]}_snr_{config["snr"]}'
    os.makedirs(result_dir, exist_ok=True)

    for cls in range(config["class_num"]):
        idx = np.where(y_true == cls)[0]
        attn_rows = [attn_list[i] for i in idx] if len(idx) else []
        kurt_rows = [kurt_list[i] for i in idx] if len(idx) else []
        attn_arr = np.vstack(attn_rows) if attn_rows else np.zeros((0, length))
        kurt_arr = np.vstack(kurt_rows) if kurt_rows else np.zeros((0, length))

        pd.DataFrame(attn_arr).to_csv(
            f"{result_dir}/attention_weights_class_{cls}.csv",
            header=False,
            index=False,
        )
        pd.DataFrame(kurt_arr).to_csv(
            f"{result_dir}/local_kurtosis_raw_class_{cls}.csv",
            header=False,
            index=False,
        )

    pd.DataFrame(y_pred_list).to_csv(f"{result_dir}/predict.csv", header=False, index=False)
    pd.DataFrame(y_true).to_csv(f"{result_dir}/truelabel.csv", header=False, index=False)

    print(f"[OK] PAttention 已导出到: {result_dir}/")


if __name__ == "__main__":
    main()

