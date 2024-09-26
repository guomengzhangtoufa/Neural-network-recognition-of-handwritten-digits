from main import Network
from torchvision import transforms
from torchvision import datasets
import torch
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    print("test_dataset length: ", len(test_dataset))

    model=Network()
    model.load_state_dict(torch.load('mnist.pth'))##加载训练好的模型

    right=0
    for i,(x,y) in enumerate(test_dataset):
        output=model(x)
        predict=output.argmax(1).item()
        if predict==y:
            right+=1
        else:
            img_path=test_dataset.samples[i][0]
            print(f"wrong case:predict={predict},y={y} img_path={img_path}")
    sample_num=len(test_dataset)
    acc=right*1.0/sample_num
    print("模型准确率为：%d/%d=%.3lf"%(right,sample_num,acc))