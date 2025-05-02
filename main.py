import torch
import torch.nn as nn
import pandas as pd
import torch.optim.adam
from torchvision import datasets, transforms


#data
raw_df = pd.read_csv('train.csv')

label = raw_df['label'].values

raw_df = raw_df.drop(['label'],axis=1)
feature = raw_df.values

train_feature = feature[:int(len(feature)*0.8)]
train_label = label[:int(len(label)*0.8)]

test_feature = feature[int(len(feature)*0.8):]
test_label = label[int(len(label)*0.8):]

train_feature = torch.tensor(train_feature).to(torch.float).cuda()
train_label = torch.tensor(train_label).cuda()
test_feature = torch.tensor(test_feature).to(torch.float).cuda()
test_label = torch.tensor(test_label).cuda()

#model structure
model = nn.Sequential(
    nn.Linear(784,444),
    nn.ReLU(),
    nn.Linear(444,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,10),
    nn.Softmax(),
).cuda()

#training

lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

for i in range(100):
    optimizer.zero_grad()
    predict = model(train_feature)
    result = torch.argmax(predict, axis=1)
    train_accurate = torch.mean((result==train_label).to(torch.float))
    loss = lossfunction(predict, train_label)
    loss.backward()
    optimizer.step()
    print('train loss:{} train_accurate:{}'.format(loss.item(), train_accurate.item()))

    optimizer.zero_grad()
    predict = model(test_feature)
    result = torch.argmax(predict, axis=1)
    test_acc = torch.mean((result == test_label).to(torch.float))
    loss = lossfunction(predict, test_label)
    print('test loss:{} test acc:{}'.format(loss.item(),test_acc.item()))

torch.save(model.state_dict(), './mymodel.pt')