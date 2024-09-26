import torch
from torch import nn

##定义神经网络Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        ##线性层1，输入层和隐藏层
        self.layer1=nn.Linear(784,256)
        ##线性层2，隐藏层和输出层
        self.layer2=nn.Linear(256,10)

        ##前向传播
    def forward(self,x):##x为输入图像
        x=x.view(-1,28*28)##将x转化成形状为（1,784）
        x=self.layer1(x)##将x输入layer1
        x=torch.relu(x)##使用relu激活
        return self.layer2(x)##输入到layer2计算结果

