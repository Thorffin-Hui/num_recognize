import torch
import torch.nn as nn
import pandas as pd
import torch.optim.adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time
import psutil

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/pure_einsum')

start_time = time.time()  # 记录开始时间

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

# model structure
# 模型结构
# model = nn.Sequential(
#     nn.Flatten(),  # 展平输入数据，将 (batch_size, 28, 28) 变为 (batch_size, 784)
#     nn.Linear(28*28, 444),  # 28*28 = 784，这里输入到第一层的大小应该是 784
#     nn.ReLU(),
#     nn.Linear(444, 512),
#     nn.ReLU(),
#     nn.Linear(512, 512),
#     nn.ReLU(),
#     nn.Linear(512, 10),
# ).to(device)

import opt_einsum  as oe

#training
W1 = torch.nn.Parameter(torch.randn(784, 444, requires_grad=True).to(device))
W2 = torch.nn.Parameter(torch.randn(444, 512, requires_grad=True).to(device))
W3 = torch.nn.Parameter(torch.randn(512, 512, requires_grad=True).to(device))
W4 = torch.nn.Parameter(torch.randn(512, 10, requires_grad=True).to(device))

lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=[W1,W2,W3,W4], lr=0.001)

for epoch in range(100):
    epoch_start_time = time.time()  # 每个epoch开始的时间
    # 记录每个 epoch 的内存和GPU占用情况
    cpu_memory_before = psutil.virtual_memory().percent  # 获取CPU内存使用百分比
    gpu_memory_before = torch.cuda.memory_allocated()  # 获取当前GPU内存占用量 (字节)

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)

        inputs=inputs.view(-1,784)
        optimizer.zero_grad()
        
        # path, path_info = oe.contract_path('ab,bc,cd,de,ef->af',inputs,W1,W2,W3,W4)
        # print(path)
        # print(path_info)
        outputs = oe.contract('ab,bc,cd,de,ef->af',inputs,W1,W2,W3,W4)
        
        loss = lossfunction(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        running_loss += loss.item()

    epoch_end_time = time.time()  # 每个epoch结束的时间
    epoch_duration = epoch_end_time - epoch_start_time
    train_accuracy = correct_train / total_train
    # 获取更新后的内存和GPU占用情况
    cpu_memory_after = psutil.virtual_memory().percent  # CPU内存使用百分比
    gpu_memory_after = torch.cuda.memory_allocated()  # GPU内存占用
     # 计算该 epoch 占用的内存变化量
    cpu_memory_usage = cpu_memory_after - cpu_memory_before  # 当前epoch CPU内存变化
    gpu_memory_usage = (gpu_memory_after - gpu_memory_before) / 1024**2  # 当前epoch GPU内存变化 (MB)
    gpu_usage = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(device).total_memory * 100

     # 记录到 TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('epoch_duration/train', epoch_duration, epoch)
    writer.add_scalar('CPU Memory Before (Percent)/train', cpu_memory_before, epoch)
    writer.add_scalar('GPU Memory Before (Percent)/train', gpu_memory_before, epoch) 
    writer.add_scalar('CPU Memory After (Percent)/train', cpu_memory_after, epoch)
    writer.add_scalar('GPU Memory After (Percent)/train', gpu_memory_after, epoch)
    writer.add_scalar('CPU Memory Usage (Percent)/train', cpu_memory_usage, epoch)
    writer.add_scalar('GPU Memory Usage (MB)/train', gpu_memory_usage, epoch) 
    writer.add_scalar('GPU Usage (%)/train', gpu_usage, epoch)
    
    print(f"Epoch {epoch + 1}/{100} - train loss: {running_loss / len(train_loader)} train accuracy: {train_accuracy}")
    
end_time = time.time()  # 记录结束时间
total_duration = end_time - start_time
print(f"Total training time: {total_duration} seconds")

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

start_time = time.time()

correct_test = 0
total_test = 0
running_test_loss = 0.0

for epoch in range(10):
    with torch.no_grad():  # No need to compute gradients during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 784)
            
            outputs = oe.contract('ab,bc,cd,de,ef->af', inputs, W1, W2, W3, W4)
            loss = lossfunction(outputs, labels)
            
            # 计算测试准确率
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            running_test_loss += loss.item()

test_accuracy = correct_test / total_test
test_loss = running_test_loss / len(test_loader)

# 记录测试信息到 TensorBoard
writer.add_scalar('Loss/test', test_loss, epoch)
writer.add_scalar('Accuracy/test', test_accuracy, epoch)

print(f"Test loss: {test_loss} test accuracy: {test_accuracy}")

writer.close()
end_time = time.time()  # 记录结束时间
total_duration = end_time - start_time
print(f"Total test time: {total_duration} seconds")