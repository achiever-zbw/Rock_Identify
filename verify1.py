import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端

# 随机种子
torch.manual_seed(42)


def dataTransforms():
    """定义验证集数据预处理"""
    val_Transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return val_Transform


def load_Dataset(dataPath, val_Transform):
    """加载验证集
    dataPath : 验证数据源
    val_Transform : 验证集数据预处理
    """
    # 加载验证集
    val_dataset = datasets.ImageFolder(dataPath, val_Transform)
    # 获取类别数量
    num_classes = len(val_dataset.classes)
    print(f"验证集共有 {num_classes} 个类别")
    print(f"验证集共 {len(val_dataset)} 张图片")
    return val_dataset, num_classes


def Model(num_classes):
    """构建模型"""
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


def predict_rock_type(model, image_path, device):
    """预测单张图片的类别
    model : 模型
    image_path : 图片路径
    device : 设备
    """
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
    """处理并保存预测结果图像
    image_path : 输入图像路径
    output_path : 输出图像路径
    predicted_class : 预测类别
    true_class : 真实类别
    classes : 类别列表
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')

    # 设置文本颜色（正确为绿色，错误为红色）
    text_color = 'green' if predicted_class == true_class else 'red'
    
    # 添加预测结果文本
    predicted_name = classes[predicted_class]
    true_name = classes[true_class]
    text = f'Pred : {predicted_name}\nTrue : {true_name}'
    
    # 添加文本背景框
    plt.text(10, 10, text, 
             color=text_color, 
             fontsize=20,
             bbox=dict(facecolor='black', alpha=0.5))

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def verify(model, val_dataset, device, output_dir):
    """验证函数
    model : 模型
    val_dataset : 验证数据集
    device : 设备
    output_dir : 输出目录
    """
    model.eval()
    correct = 0
    total = 0
    
    # 创建结果保存目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取类别名称
    class_names = val_dataset.classes
    
    # 遍历验证集
    for idx, (inputs, label) in enumerate(val_dataset):
        inputs = inputs.unsqueeze(0).to(device)
        label = torch.tensor(label).to(device)  # 将标签转换为tensor
        
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计正确预测数
            total += 1
            if predicted.item() == label.item():
                correct += 1
            
            # 获取类别名称
            predicted_name = class_names[predicted.item()]
            true_name = class_names[label.item()]
            
            # 处理并保存图像
            img_path = val_dataset.samples[idx][0]  # 获取原始图像路径
            output_path = os.path.join(output_dir, f"{true_name}_{os.path.basename(img_path)}")
            process_image(img_path, output_path, predicted.item(), label.item(), class_names)
            
            # 打印预测结果
            flag = True
            if predicted_name != true_name:
                flag = False
            print(f"图片 {idx+1}:  预测类型:{predicted_name}   实际类型:{true_name}  {flag}")
    
    # 计算并打印准确率
    accuracy = 100 * correct / total
    print(f"\n验证集准确率: {accuracy:.2f}%")
    print(f"总图片数: {total}")
    print(f"正确预测数: {correct}")


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置数据路径
    val_path = r"D:\裂缝数据集\数据集2\val"
    
    # 数据准备
    val_transform = dataTransforms()
    val_dataset, num_classes = load_Dataset(val_path, val_transform)
    output_dir = r"D:\裂缝数据集\result"
    # 构建模型
    model = Model(num_classes)
    model = model.to(device)
    
    # 加载模型权重（使用weights_only=True避免安全警告）
    model.load_state_dict(torch.load('final_model.pth', weights_only=True))
    print("模型加载完成")
    
    # 开始验证
    verify(model, val_dataset, device, output_dir)
