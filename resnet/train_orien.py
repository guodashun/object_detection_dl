import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
from resnet import ResNetXS
import torchvision.transforms as transforms
# import argparse
import os
from dataset import RobotTeamClassification
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "./data"
item = "orientation"
param = "/param8"
save_folder = "./weights/" + item + param
log_dir = "./log/" + item + param
batch_size = 128
num_workers = 2
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1

start_iter = 0
max_epoch = 3000
adjust_step = []

transform_train = transforms.Compose([
    transforms.CenterCrop(90),
    # transforms.Resize(31),
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    # transforms.ColorJitter(brightness=1, contrast=1),
    transforms.CenterCrop(90),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = RobotTeamClassification(root=dataset_root, transform=transform_train, phase="train", item=item)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = RobotTeamClassification(root=dataset_root, transform=transform_test, phase="test", item=item)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).to(device)
net = ResNetXS(18, True).to(device)


writer = SummaryWriter(log_dir=log_dir)


def train():
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # best_acc = 40
    best_loss = 9999
    print("Start Training!")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    for epoch in range(start_iter, max_epoch):
        print('\nEpoch: %d' % (epoch + 1))
        if epoch in adjust_step:
            adjust_lr(optimizer)
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.view(-1, 1)
            # print("output:", outputs.shape, "label:", labels.shape)
            # outputs = outputs.float()
            # labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            # writer.add_scalar("train_acc", 100. * correct / total, epoch + 1)
        writer.add_scalar("loss", sum_loss / (i + 1), epoch + 1)

        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            sum_loss = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                labels = labels.view(-1, 1)
                test_loss = criterion(outputs, labels)
                writer.add_scalar("test_loss", test_loss.item() / len(testset), epoch + 1)
                if test_loss.item() < best_loss:
                    torch.save(net.state_dict(), '%s/best_net.pth' % (save_folder))
                    best_loss = test_loss
                _, predicted = torch.max(outputs.data, 1)
                sum_loss += test_loss.item()
            print('测试分类Loss为：%.3f' % (sum_loss / len(testset)))
            if epoch % 30 == 0 and epoch != 0:
                print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (save_folder, epoch))

    print("Training Finished, TotalEPOCH=%d" % max_epoch)


def adjust_lr(optimizer):
    learning_rate = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


if __name__ == "__main__":
    train()
