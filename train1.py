import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
import torch.optim as optim
import torch.nn as nn
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP问题

matplotlib.use('TkAgg')  # 使用 TkAgg 后端

# 随机种子
torch.manual_seed(42)
np.random.seed(42)


def dataTransforms():
    """定义 训练集数据预处理 与 """
    train_Transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 调整裁剪范围
        transforms.RandomRotation(45),                        # 增加旋转角度
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 增加平移范围
        transforms.ToTensor(),  # 先转换为张量
        transforms.RandomErasing(p=0.5, scale=(
            0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    val_Transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    return train_Transform, val_Transform


def load_Dataset(dataPath, train_Transform, val_Transform):
    """加载数据集
    dataPath : 训练数据源
    train_Transform : 训练集数据预处理
    val_Transform : 验证集数据预处理
    """
    # 加载训练集
    AllDataset = datasets.ImageFolder(dataPath, train_Transform)
    # 获取类别数量 （看看有没有错误，和实际类别对比一下下）
    num_classes = len(AllDataset.classes)
    print(f"一共有 {num_classes} 个训练类别")

    # 按照 8：2 的比例划分训练集和验证集
    trainSize = int(0.8 * len(AllDataset))
    valSize = len(AllDataset) - trainSize
    print(f"训练集共 {trainSize} 张图片，验证集共 {valSize} 张图片")
    trainDataset, valDataset = random_split(AllDataset, [trainSize, valSize])

    return trainDataset, valDataset, num_classes


def Model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # 冻结卷积层
    for param in model.parameters():
        param.requires_grad = False

    in_Features = model.fc.in_features  # 先把前一层的输入调出来
    # 最后由于输出类别较少，所以进行修改使收敛更平缓一些
    model.fc = nn.Sequential(
        nn.Linear(in_Features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model


def train(model, dataLoaders, LossFunc, optimizer, scheduler, num_epoches):
    """训练函数
    model : 模型
    dataloader : 数据加载集
    LossFunc : 损失函数
    optimizer : 优化器
    scheduler : 学习率调度器
    num_epoches : 训练次数
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    losses = []  # 记录每次训练的损失
    # 开始训练
    for epoch in range(num_epoches):
        model.train()
        runningCorrects = 0
        runningLoss = 0
        for inputs, labels in dataLoaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # 获取输出
            _, preds = torch.max(outputs, 1)
            loss = LossFunc(outputs, labels)  # 获取损失

            loss.backward()  # 反向传播
            optimizer.step()
            scheduler.step()  # 每个batch更新学习率

            runningCorrects += torch.sum(preds == labels.data)  # 计累计正确的数量
            runningLoss += loss.item() * inputs.size(0)  # inputs.size(0) : 批次的数量

        trainLoss = runningLoss / len(dataLoaders["train"].dataset)
        trainAcc = 100.*runningCorrects.double() / \
            len(dataLoaders["train"].dataset)

        # 验证阶段
        model.eval()
        runningCorrects = 0
        runningLoss = 0
        with torch.no_grad():
            for inputs, labels in dataLoaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = LossFunc(outputs, labels)
                runningCorrects += torch.sum(preds == labels.data)
                runningLoss += loss.item()*inputs.size(0)
        valLoss = runningLoss/len(dataLoaders["val"].dataset)
        valAcc = 100.*runningCorrects.double() / \
            len(dataLoaders["val"].dataset)
        # 记录损失
        losses.append(trainLoss)
        # 打印训练日志
        print(f"第 {epoch+1}/{num_epoches} 次训练 --> 训练损失值: {trainLoss:.4f} 准确率: {trainAcc:.2f}% | 验证损失值: {valLoss:.4f} 准确率: {valAcc:.2f}%")

    torch.save(model.state_dict(), "final_model.pth")
    print("训练结束，已保存模型")
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epoches + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_2.png')
    print("损失曲线图已保存")
    return model


if __name__ == '__main__':
    # 各项参数设置
    dataPath = r"D:\岩石种类识别\数据集2\train"
    batch_size = 32  # 增加batch_size
    num_epoches = 200
    # 数据准备阶段
    trainTransform, valTransform = dataTransforms()  # 初始化数据预处理方式
    trainDataset, valDataset, num_classes = load_Dataset(
        dataPath, trainTransform, valTransform)
    dataloader = {"train": DataLoader(trainDataset, batch_size=batch_size, shuffle=True),
                  "val": DataLoader(valDataset, batch_size=batch_size, shuffle=False)
                  }
    # 构建模型阶段
    model = Model(num_classes)
    # 损失函数与优化器、学习率调度器
    LossFunc = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epoches,
        steps_per_epoch=len(dataloader['train']),
        pct_start=0.3,  # 预热阶段占总训练时间的比例
        div_factor=25,  # 初始学习率与最大学习率的比值
        final_div_factor=1e4  # 最终学习率与最大学习率的比值
    )

    # 模型训练阶段
    model = train(
        model=model,
        dataLoaders=dataloader,
        LossFunc=LossFunc,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epoches=num_epoches
    )
