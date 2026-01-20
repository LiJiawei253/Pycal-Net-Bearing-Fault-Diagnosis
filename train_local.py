import os
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import  confusion_matrix, recall_score, \
    precision_score, f1_score
from utils.JNUProcessing import JNU_Processing
from utils.PUProcessing import Paderborn_Processing
from torch import nn
from torch.utils.data import DataLoader
from Model.BDCNN import BDWDCNN



from utils.DatasetLoader import CustomTensorDataset
import matplotlib.pyplot as plt # Import matplotlib




use_gpu = torch.cuda.is_available()

def UW(losses):
    loss_scale = nn.Parameter(torch.tensor([-0.5] * 3)).cuda()
    loss = (losses / (3 * loss_scale.exp()) + loss_scale / 3).sum()
    return loss


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def select_model(config):
    # 只支持 bdcnn 模型（其他模型已移除）
    if config["chosen_model"] == 'bdcnn':
        if 'pulse_attention' in config:
            model = BDWDCNN(
                config['class_num'],
                pulse_config=config['pulse_attention'],
                use_pag=config.get('use_pag', True)
            )
        else:
            model = BDWDCNN(
                config['class_num'],
                use_pag=config.get('use_pag', True)
            )
    else:
        raise ValueError(f"Unsupported model: {config['chosen_model']}. Only 'bdcnn' is supported.")
    return model


def get_checkpoint_path(config):
    """
    Allow custom checkpoint naming so ablation runs (e.g., no-PAG) do not overwrite
    the default PDIF checkpoint.
    """
    name = config.get('checkpoint_name', f'{config["dataset"]}_best_checkpoint_{config["chosen_model"]}.pth')
    # 保持向后兼容：如果用户没有带扩展名，自动补上
    if not name.endswith('.pth'):
        name = f'{name}.pth'
    return os.path.join('Models', name)



def train(config, dataloader):
    net = select_model(config)
    if use_gpu:
        net.cuda()
    train_loss = []
    train_acc = []
    valid_acc = []
    max_acc = 0
    
    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-8
    )
    loss_func = nn.CrossEntropyLoss()

    for e in range(config['epochs']):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0
            if phase == 'train':
                net.train()
            if phase == 'validation':
                net.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()
                y_hat, k, g, feat_qcnn, pag_mask = net(x)
                classifyloss = loss_func(y_hat, y)
                losses = torch.zeros(3, device=y.device)
                losses[0], losses[1], losses[2] = classifyloss, k, g

                # 计算总损失（已移除一致性损失）
                loss = UW(losses)
                if phase == 'train':
                    optimizer.zero_grad()
                    # loss = uw.forward(losses)
                    loss.backward()
                    optimizer.step()
                loss_total += loss.item()

                y_predict = y_hat.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    correct += (y_predict == y).cpu().squeeze().sum().numpy()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()

                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                          e + 1, step + 1, len(dataloader[phase].dataset), loss_total))

            acc = correct / total
            if phase == 'train':
                train_loss.append(loss_total)
                train_acc.append(acc)

            if phase == 'validation':
                valid_acc.append(acc)
                if acc > max_acc:
                    max_acc = acc
                    if not os.path.exists("Models"):
                        os.mkdir('Models')
                    # 存储模型参数
                    torch.save(net.state_dict(), get_checkpoint_path(config))
                    print("save model")
            print('%s ACC:%.4f' % (phase, acc))
        scheduler.step()
    return net


def inference(dataloader, config):
    net = select_model(config)
    # Check if model checkpoint exists before loading
    checkpoint_path = get_checkpoint_path(config)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return None, None, None # Return None for features, labels, and metrics

    state_dict = torch.load(checkpoint_path)
    net.load_state_dict(state_dict)
    y_list, y_predict_list = [], []
    features_list = [] # List to store extracted features

    if use_gpu:
        net.cuda()
    net.eval()

    # Define a hook to capture features from the desired layer
    # In BDWDCNN, the last convolutional features are the output of self.cnn
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()
        return hook

    # Register the hook to the last layer of self.cnn
    # Assuming self.cnn is a Sequential model and we want the output of its last module
    if hasattr(net, 'cnn') and len(net.cnn._modules) > 0:
        last_cnn_module_name = list(net.cnn._modules.keys())[-1]
        handle = net.cnn._modules[last_cnn_module_name].register_forward_hook(get_features('cnn_output'))
    else:
        print("Warning: Model does not have a 'cnn' attribute or it is empty. Feature extraction may fail.")
        handle = None


    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()

            # Forward pass - the hook will capture the features
            outputs = net(x)
            y_hat = outputs[0] if isinstance(outputs, tuple) else outputs

            # Append the captured features and labels
            if 'cnn_output' in features:
                 # Global average pooling might be needed depending on the layer output shape
                 # Let's flatten the features for t-SNE
                 flattened_features = features['cnn_output'].view(features['cnn_output'].size(0), -1)
                 features_list.append(flattened_features)
                 del features['cnn_output'] # Clear the captured features for the next batch


            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())

    # Remove the hook if it was registered
    if handle:
        handle.remove()

    # Concatenate features from all batches
    all_features = torch.cat(features_list, dim=0).numpy() if features_list else np.array([])
    all_labels = np.array(y_list)


    # Calculate metrics (optional, but good to keep)
    if all_labels.size > 0:
        cnf_matrix = confusion_matrix(all_labels, y_predict_list)
        recall = recall_score(all_labels, y_predict_list, average="macro")
        precision = precision_score(all_labels, y_predict_list, average="macro")
        F1 = f1_score(all_labels, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        metrics = {
            "F1 Score": F1,
            "FPR": FPR,
            "Recall": recall,
            'PRE': precision
        }
        print("Inference Metrics:")
        print(metrics)
        # 可视化混淆矩阵
        save_path = plot_confusion_matrix(all_labels, y_predict_list, config, save_dir='results')
    else:
        metrics = None
        print("No data processed for inference.")


    return all_features, all_labels, metrics # Return features and labels


def plot_confusion_matrix(y_true, y_pred, config, save_dir='results'):
    """
    绘制并保存混淆矩阵
    
    Args:
        y_true: 真实标签 (numpy array)
        y_pred: 预测标签 (numpy array)
        config: 配置字典，包含模型和数据集信息
        save_dir: 保存目录
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算混淆矩阵
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap='Blues', vmin=0, vmax=cnf_matrix.max())
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            plt.text(j, i, int(cnf_matrix[i, j]),
                     ha="center", va="center",
                     color="white" if cnf_matrix[i, j] > (cnf_matrix.max() * 0.5) else "black",
                     fontsize=8)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 生成标题
    model_name = config.get('chosen_model', 'Model')
    dataset_name = config.get('dataset', 'Dataset')
    snr = config.get('snr', 'N/A')
    title = f'Confusion Matrix - {model_name.upper()} on {dataset_name} (SNR={snr}dB)'
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    # 生成文件名（包含时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_{model_name}_{dataset_name}_SNR{snr}dB_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 混淆矩阵已保存至: {save_path}")
    plt.close()  # 关闭图形，避免在服务器上显示
    
    return save_path


def main(config):
    random_seed(config['seed'])

    data_root = config.get('data_root', 'data')

    if config['dataset'] == "Paderborn":
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = Paderborn_Processing(file_path=os.path.join(data_root, config['dataset']), load=config['chosen_dataset'], noise=config['add_noise'], snr=config['snr'])
        config['class_num'] = 14

    elif config['dataset'] == 'JNU':
        # Set class_num for JNU dataset
        config['class_num'] = 10
        Train_X, Val_X, Test_X, Train_Y, Val_Y, Test_Y = JNU_Processing(file_path=os.path.join(data_root, config['dataset']),noise=config['add_noise'], snr=config['snr'])


    train_dataset = CustomTensorDataset(Train_X, Train_Y)
    valid_dataset = CustomTensorDataset(Val_X, Val_Y)
    test_dataset = CustomTensorDataset(Test_X, Test_Y)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 96), shuffle=False, drop_last=False)

    # Original training and inference flow
    print("Running training and inference...")
    train(config, data_loaders)
    inference(test_loader, config)



if __name__ == '__main__':
    config = {'seed': 42,
              'batch_size': 96,#96
              'epochs': 200,
              'lr': 0.5,
              'add_noise': 'Gaussian', # Gaussian, pink, Laplace, airplane, truck
              'snr': -4, #dB
              'dataset': 'Paderborn', # Paderborn, JNU
              'chosen_dataset': 'N09_M07_F10', # N09_M07_F10; N15_M01_F10; N15_M07_F04; N15_M07_F10;
              'chosen_model': 'bdcnn', 
              'class_num': 14,  # default, will be updated based on dataset
              'pulse_attention': {
                  'multiscale': True,         # 是否使用多尺度峭度计算
                  'kernel_sizes': [15, 31, 63], 
                  'threshold':3.0         
              }
            }
    main(config)






