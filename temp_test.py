import torch

# 定义设备（CPU 或 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 创建叶子张量并设置为可训练参数
W1 = torch.nn.Parameter(torch.randn(784, 444, requires_grad=True).to(device))
W2 = torch.nn.Parameter(torch.randn(444, 512, requires_grad=True).to(device))
W3 = torch.nn.Parameter(torch.randn(512, 512, requires_grad=True).to(device))
W4 = torch.nn.Parameter(torch.randn(512, 10, requires_grad=True).to(device))

# 确认这些参数是叶子张量
print(W1.is_leaf)  # 应该输出 True
print(W2.is_leaf)  # 应该输出 True
print(W3.is_leaf)  # 应该输出 True
print(W4.is_leaf)  # 应该输出 True

# 创建优化器
optimizer = torch.optim.Adam(params=[W1, W2, W3, W4], lr=0.001)
