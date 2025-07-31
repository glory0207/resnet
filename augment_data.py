import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

def advanced_augmentation(image_path, output_dir, num_augmentations=5):
    """对单张图片进行高级数据增强"""
    img = Image.open(image_path).convert('RGB')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    augmented_images = []
    
    for i in range(num_augmentations):
        # 复制原始图像
        aug_img = img.copy()
        
        # 1. 随机亮度调整
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # 2. 随机对比度调整
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # 3. 随机色彩饱和度
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # 4. 随机锐度调整
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # 5. 随机模糊
        if random.random() > 0.7:
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # 6. 随机噪声
        if random.random() > 0.6:
            img_array = np.array(aug_img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            aug_img = Image.fromarray(img_array)
        
        # 7. 随机旋转
        if random.random() > 0.4:
            angle = random.uniform(-15, 15)
            aug_img = aug_img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        # 8. 随机缩放和裁剪
        if random.random() > 0.4:
            scale = random.uniform(0.9, 1.1)
            w, h = aug_img.size
            new_w, new_h = int(w * scale), int(h * scale)
            aug_img = aug_img.resize((new_w, new_h))
            
            # 裁剪回原始大小
            if scale > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                aug_img = aug_img.crop((left, top, left + w, top + h))
            else:
                # 填充到原始大小
                new_img = Image.new('RGB', (w, h), (128, 128, 128))
                paste_x = (w - new_w) // 2
                paste_y = (h - new_h) // 2
                new_img.paste(aug_img, (paste_x, paste_y))
                aug_img = new_img
        
        # 保存增强后的图像
        output_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.jpg")
        aug_img.save(output_path, quality=95)
        augmented_images.append(output_path)
    
    return augmented_images

def balance_dataset(data_dir, target_samples_per_class=50):
    """平衡数据集，为样本少的类别生成更多数据"""
    
    # 统计每个类别的样本数
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
            print(f"{class_name}: {count} 样本")
    
    # 为样本少的类别生成增强数据
    for class_name, count in class_counts.items():
        if count < target_samples_per_class:
            print(f"\n为类别 '{class_name}' 生成增强数据...")
            class_path = os.path.join(data_dir, class_name)
            
            # 获取该类别的所有图片
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 计算需要生成的增强样本数
            needed_samples = target_samples_per_class - count
            augmentations_per_image = max(1, needed_samples // len(image_files))
            
            generated_count = 0
            for img_file in image_files:
                if generated_count >= needed_samples:
                    break
                    
                img_path = os.path.join(class_path, img_file)
                try:
                    # 生成增强样本
                    aug_count = min(augmentations_per_image, needed_samples - generated_count)
                    augmented = advanced_augmentation(img_path, class_path, aug_count)
                    generated_count += len(augmented)
                    print(f"  为 {img_file} 生成了 {len(augmented)} 个增强样本")
                except Exception as e:
                    print(f"  处理 {img_file} 时出错: {e}")
            
            print(f"类别 '{class_name}' 总共生成了 {generated_count} 个增强样本")
    
    # 重新统计增强后的样本数
    print("\n增强后的数据集统计:")
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{class_name}: {count} 样本")

def create_additional_validation_data(train_dir, val_dir, validation_ratio=0.2):
    """从训练集中分出一部分作为验证集，特别是样本少的类别"""
    
    for class_name in os.listdir(train_dir):
        train_class_path = os.path.join(train_dir, class_name)
        val_class_path = os.path.join(val_dir, class_name)
        
        if not os.path.isdir(train_class_path):
            continue
            
        # 确保验证集目录存在
        os.makedirs(val_class_path, exist_ok=True)
        
        # 获取训练集中该类别的所有图片
        train_images = [f for f in os.listdir(train_class_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'aug' in f]
        
        # 计算需要移动到验证集的数量
        val_count = max(3, int(len(train_images) * validation_ratio))  # 至少3张
        
        if val_count > 0 and len(train_images) > val_count:
            # 随机选择图片移动到验证集
            selected_images = random.sample(train_images, val_count)
            
            for img_name in selected_images:
                src_path = os.path.join(train_class_path, img_name)
                dst_path = os.path.join(val_class_path, img_name)
                shutil.move(src_path, dst_path)
            
            print(f"为类别 '{class_name}' 移动了 {len(selected_images)} 张图片到验证集")

if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/val"
    
    print("开始数据增强...")
    
    # 1. 平衡训练集
    balance_dataset(train_dir, target_samples_per_class=60)
    
    # 2. 为验证集增加样本
    print(f"\n为验证集增加样本...")
    create_additional_validation_data(train_dir, val_dir, validation_ratio=0.15)
    
    print("\n数据增强完成！")
