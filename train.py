import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from dataset import get_dataloaders
from model import create_model

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
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
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据路径
    train_dir = 'data/train'
    val_dir = 'data/val'
    
    # 获取数据加载器
    train_loader, val_loader, classes = get_dataloaders(train_dir, val_dir, batch_size=32)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 创建模型
    model = create_model(num_classes=len(classes))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 分阶段训练
    # 阶段1：冻结backbone，只训练分类器
    print("Stage 1: Training classifier only")
    optimizer = optim.Adam(model.backbone.fc.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model, _, _, _, _ = train_model(model, dataloaders, criterion, optimizer, scheduler, 
                                 num_epochs=10, device=device)
    
    # 阶段2：解冻部分层进行fine-tuning
    print("\nStage 2: Fine-tuning")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=15, device=device)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'mobile_screen_classifier.pth')
    
    print("Model saved as 'mobile_screen_classifier.pth'")

if __name__ == '__main__':
    main()