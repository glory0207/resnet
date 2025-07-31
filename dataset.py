import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from collections import Counter

class MobileScreenDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentation_factor=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['加载不全', '弹窗', '桌面页', '登录页', '空白页']  # 修正了类别名称
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.augmentation_factor = augmentation_factor
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # 应用数据增强策略
        if self.augmentation_factor:
            self._apply_augmentation_strategy()
    
    def _apply_augmentation_strategy(self):
        """对样本少的类别进行数据增强"""
        # 统计每个类别的样本数
        class_counts = Counter([sample[1] for sample in self.samples])
        max_count = max(class_counts.values())
        
        # 为样本少的类别增加样本
        augmented_samples = []
        for class_idx, count in class_counts.items():
            if count < max_count * 0.8:  # 如果样本数少于最大类别的80%
                class_samples = [s for s in self.samples if s[1] == class_idx]
                # 计算需要增强的倍数
                target_count = int(max_count * 0.8)
                multiplier = target_count // count
                for _ in range(multiplier - 1):  # -1 因为原始样本已经存在
                    augmented_samples.extend(class_samples)
        
        self.samples.extend(augmented_samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    # 更强的数据增强用于样本少的类别
    strong_augment_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # 增加垂直翻转
        transforms.RandomRotation(degrees=15),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 仿射变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # 随机擦除
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform, strong_augment_transform

def get_dataloaders(train_dir, val_dir, batch_size=32, use_balanced_sampling=True):
    train_transform, val_transform, strong_augment_transform = get_transforms()
    
    # 对训练集应用数据增强策略
    train_dataset = MobileScreenDataset(train_dir, transform=train_transform, augmentation_factor=True)
    val_dataset = MobileScreenDataset(val_dir, transform=val_transform)
    
    # 创建平衡采样器
    train_loader = None
    if use_balanced_sampling:
        # 计算类别权重
        class_counts = Counter([sample[1] for sample in train_dataset.samples])
        total_samples = len(train_dataset.samples)
        class_weights = [total_samples / class_counts[i] for i in range(len(train_dataset.classes))]
        
        # 为每个样本分配权重
        sample_weights = [class_weights[sample[1]] for sample in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 打印类别分布信息
    print("训练集类别分布:")
    class_counts = Counter([sample[1] for sample in train_dataset.samples])
    for class_name, class_idx in train_dataset.class_to_idx.items():
        print(f"  {class_name}: {class_counts[class_idx]} 样本")
    
    return train_loader, val_loader, train_dataset.classes