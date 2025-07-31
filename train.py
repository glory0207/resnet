import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
from collections import Counter
from dataset import get_dataloaders
from model import create_model

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(dataset):
    """计算类别权重"""
    class_counts = Counter([sample[1] for sample in dataset.samples])
    total_samples = len(dataset.samples)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0  # 添加F1分数跟踪
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 收集预测和标签用于计算详细指标
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # 计算每个类别的精确率和召回率
            from sklearn.metrics import classification_report, f1_score
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {f1:.4f}')
            
            if phase == 'val':
                # 打印详细的分类报告
                class_names = ['加载不全', '弹窗', '桌面页', '登录页', '空白页']
                print("\n分类报告:")
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
            # 使用F1分数作为模型选择标准
            if phase == 'val' and f1 > best_f1:
                best_f1 = f1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best val F1: {best_f1:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据路径
    train_dir = 'data/train'
    val_dir = 'data/val'
    
    # 获取数据加载器（使用平衡采样）
    train_loader, val_loader, classes = get_dataloaders(
        train_dir, val_dir, batch_size=32, use_balanced_sampling=True
    )
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 创建模型
    model = create_model(num_classes=len(classes))
    model = model.to(device)
    
    # 计算类别权重
    class_weights = calculate_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)
    
    # 使用加权交叉熵损失或Focal Loss
    use_focal_loss = True
    if use_focal_loss:
        alpha = class_weights / class_weights.sum()  # 归一化权重
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        print("使用 Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用加权交叉熵损失")
    
    print(f"类别权重: {class_weights}")
    
    # 分阶段训练
    # 阶段1：冻结backbone，只训练分类器
    print("\nStage 1: Training classifier only")
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model, _, _, _, _ = train_model(model, dataloaders, criterion, optimizer, scheduler, 
                                 num_epochs=15, device=device)  # 增加训练轮数
    
    # 阶段2：解冻部分层进行fine-tuning
    print("\nStage 2: Fine-tuning")
    model.unfreeze_all()  # 解冻所有层
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.0001, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=20, device=device)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'class_weights': class_weights.cpu(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'mobile_screen_classifier.pth')
    
    print("Model saved as 'mobile_screen_classifier.pth'")

if __name__ == '__main__':
    main()