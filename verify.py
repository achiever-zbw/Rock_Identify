import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import numpy as np


def predict_rock_type(model, image_path, device):
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs).item()

    return predicted_class


def process_image(image_path, output_path, predicted_class, true_class, classes):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # 设置文本颜色（正确为绿色，错误为红色）
    text_color = 'green' if predicted_class == true_class else 'red'
    
    # 添加预测结果文本
    predicted_name = classes[predicted_class]
    true_name = classes[true_class]
    text = f'预测类型: {predicted_name}\n实际类型: {true_name}'
    
    # 添加文本背景框
    plt.text(10, 10, text, 
             color=text_color, 
             fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top')

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_classes_from_train(data_dir):
    """从训练集获取类别信息"""
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    return train_dataset.classes


def build_model(num_classes):
    """构建模型"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结卷积层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换全连接层
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置数据目录
    data_dir = r'D:\岩石种类识别\数据集2'
    
    # 获取类别信息
    classes = get_classes_from_train(data_dir)
    num_classes = len(classes)
    print(f'检测到 {num_classes} 个类别')
    print(f'类别列表: {classes}')

    # 加载模型
    model = build_model(num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load('final_model.pth', weights_only=True))  # 加载最佳模型
    model.eval()

    # 设置验证集目录和输出目录
    verify_dir = os.path.join(data_dir, 'val')
    output_dir = r'D:\岩石种类识别\result'
    os.makedirs(output_dir, exist_ok=True)

    # 统计变量
    total_images = 0
    correct_predictions = 0

    # 遍历验证集目录
    for class_folder in os.listdir(verify_dir):
        class_path = os.path.join(verify_dir, class_folder)
            
        # 获取实际类别索引
        true_class = classes.index(class_folder)  
        
        # 处理该类别下的所有图片
        for img_name in os.listdir(class_path):      
            img_path = os.path.join(class_path, img_name)
            predicted_class = predict_rock_type(model, img_path, device)

            total_images += 1
            if predicted_class == true_class:
                correct_predictions += 1
            
            # 处理并保存图像
            output_path = os.path.join(output_dir, f"{class_folder}_{img_name}")
            process_image(img_path, output_path, predicted_class, true_class, classes)
            
            # 打印预测结果
            print(f"图片: {img_name}")
            print(f"预测类型: {classes[predicted_class]}")
            print(f"实际类型: {classes[true_class]}")
            print("-" * 50)

    # 打印总体准确率
    accuracy = correct_predictions / total_images
    print(f"\n总体准确率: {accuracy:.2%}")
    print(f"总图片数: {total_images}")
    print(f"正确预测数: {correct_predictions}")


if __name__ == "__main__":
    main()
