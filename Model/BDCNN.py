import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.ClassBD_Module import CLASSBD
from Model.ConvQuadraticOperation import ConvQuadraticOperation




class BDWDCNN(nn.Module):
    """
    Quadraic in Attention CNN
    """

    def __init__(self, n_classes, pulse_config=None, use_pag=True) -> object:
        super(BDWDCNN, self).__init__()

        # 使用配置创建CLASSBD实例，传递脉冲注意力参数
        if pulse_config is None:
            # 默认配置，向后兼容
            self.classbd = CLASSBD(use_pag=use_pag)
        else:
            # 创建带有自定义脉冲注意力配置的CLASSBD实例
            self.classbd = CLASSBD(pulse_config=pulse_config, use_pag=use_pag)

        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1D_1', nn.Conv1d(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))




        self.__make_layer(16, 32, 1, 2)
        self.__make_layer(32, 64, 1, 3)   #  改64
        self.__make_layer(64, 64, 1, 4)   #  改64
        self.__make_layer(64, 64, 1, 5)   #  改64
        self.__make_layer(64, 64, 0, 6)
        self.fc1 = nn.Linear(192, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, n_classes)

    def __make_layer(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))




    def forward(self, x):

        a2, k, g, feat_qcnn, pag_mask = self.classbd(x)
        # backbone
        out = self.cnn(a2)
        out = self.fc1(out.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1), k, g, feat_qcnn, pag_mask


    


 
