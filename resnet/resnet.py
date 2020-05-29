import torch.nn as nn
import torchvision


class ResNetX(nn.Module):
    def __init__(self, resnet_x, pretrained=True, num_classes=1000):
        super(ResNetX, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        if resnet_x == 18:
            model = torchvision.models.resnet18(pretrained=self.pretrained)
        elif resnet_x == 50:
            model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif resnet_x == 101:
            model = torchvision.models.resnet101(pretrained=self.pretrained)
        # modules = list(model.children())[:-1]
        # self.resnet_18_pre = nn.Sequential(*modules)  # 加星号为了将每一层拆开变成单个元素
        x = model.fc.in_features
        # print(x)
        model.fc = nn.Linear(x, num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class ResNetXS(nn.Module):
    def __init__(self, resnet_x, pretrained=True):
        super(ResNetXS, self).__init__()
        self.pretrained = pretrained
        if resnet_x == 18:
            model = torchvision.models.resnet18(pretrained=self.pretrained)
        elif resnet_x == 50:
            model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif resnet_x == 101:
            model = torchvision.models.resnet101(pretrained=self.pretrained)
        # modules = list(model.children())[:-1]
        # self.resnet_18_pre = nn.Sequential(*modules)  # 加星号为了将每一层拆开变成单个元素
        x = model.fc.in_features
        # print(x)
        model.fc = nn.Linear(x, 1)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # print(out.shape, out)
        # 输出为所有batch_size的结果 shape为batch_size×num_classes
        return out
