import torch
import torch.nn as nn
import pandas as pd
import torch.optim.adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设备设置（自动使用CUDA，如果不可用则回退到CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用DataLoader将数据分批处理
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#model structure
# 模型结构
model = nn.Sequential(
    nn.Flatten(),  # 展平输入数据，将 (batch_size, 28, 28) 变为 (batch_size, 784)
    nn.Linear(28*28, 444),  # 28*28 = 784，这里输入到第一层的大小应该是 784
    nn.ReLU(),
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(device)


#training

lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(100):

    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfunction(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        running_loss += loss.item()

    train_accuracy = correct_train / total_train
    print(f"Epoch {epoch + 1}/{100} - train loss: {running_loss / len(train_loader)} train accuracy: {train_accuracy}")

torch.save(model.state_dict(), './mymodel.pt')