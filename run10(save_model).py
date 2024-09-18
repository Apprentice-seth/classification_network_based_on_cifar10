import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def get_trainset_loader(self):
        to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root="", train=True, transform=to_tensor, download=False)
        return DataLoader(trainset, batch_size=16, shuffle=True)

    def get_testset_loader(self):
        to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root="", train=False, transform=to_tensor, download=False)
        return DataLoader(testset, batch_size=16, shuffle=True)

    @staticmethod
    def evaluate(test_data, net):
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for (x, y) in test_data:
                outputs = net(x)
                preds = torch.argmax(outputs, dim=1)
                n_correct += (preds == y).sum().item()
                n_total += y.size(0)
        return n_correct / n_total

def main():
    net = Net()
    train_data_loader = net.get_trainset_loader()
    test_data_loader = net.get_testset_loader()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("initial accuracy:", net.evaluate(test_data_loader, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(3):
        for (x, y) in train_data_loader:
            net.zero_grad()
            output = net.forward(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", net.evaluate(test_data_loader, net))

    # 使用state_dict保存模型参数
    torch.save(net.state_dict(), r'G:\CIFAR10\model\model.pt')

    for (n, (x, _)) in enumerate(test_data_loader):
        if n > 9:
            break
        predict = torch.argmax(net.forward(x), dim=1)
        plt.figure(n)
        plt.imshow(x[0].permute(1, 2, 0).numpy())
        plt.title("prediction: " + class_names[predict[0].item()])
        plt.savefig(f"results/{class_names[predict[0].item()]}.png")  # 保存图片（可选）
    plt.show()




if __name__ == "__main__":
    main()
