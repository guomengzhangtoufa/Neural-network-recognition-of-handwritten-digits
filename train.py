import torch
from torch import nn
from torch import optim
from main import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import time

start_time=time.time()
if __name__=='__main__':
 #图像预处理
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor()])##转换为单通道灰度图并转化为张量

 #读入并构造数据集
train_dataset=datasets.ImageFolder(root='./mnist_images/train',transform=transform)
print("train_dataset_length: ",len(train_dataset))
##shuffle = True：设置为True表示在每个训练轮次（epoch）开始时，DataLoader会随机打乱train_dataset中的数据顺序。
# 这样做的目的是为了避免模型在训练过程中对数据的顺序产生依赖，从而提高模型的泛化能力。
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

model = Network()
optimizer=optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()##使用交叉熵损失函数

for epoch in range(10):
    for batch_idx, (data, label) in enumerate(train_loader):
        output=model(data)
        loss=criterion(output,label)##label是真实值，output是预测值
        loss.backward()#反向传播计算梯度
        optimizer.step()##更新参数
        optimizer.zero_grad()#梯度清零

        if batch_idx%1000==0:
            print(f"Epoch{epoch+1}/10"
                  f"| Batch {batch_idx}/{len(train_loader)}"
                  f"| Loss: {loss.item():.4f}")##这个张量中提取出一个 Python 标量值
torch.save(model.state_dict(),'mnist.pth')##保存模型
end_time=time.time()
time=start_time-end_time
print("总耗时： ",time)

