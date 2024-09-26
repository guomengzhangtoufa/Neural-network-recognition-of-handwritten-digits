from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
##该模块作为主程序入口运行时执行，而当这个模块被其他模块导入时，这段代码将不会被执行。
if __name__=='__main__':
 #图像预处理
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor()])##转换为单通道灰度图并转化为张量
train_dataset=datasets.ImageFolder(root='./mnist_images/train',transform=transform)
test_dataset=datasets.ImageFolder(root='./mnist_images/test',transform=transform)
print("train_dataset length: ",len(train_dataset))
print("test_dataset length: ",len(test_dataset))

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
print("train_loader length: ",len(train_loader))