#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重命名图片文件为img1、img2...格式
"""

import os
import shutil
from pathlib import Path

def rename_images_in_folder(folder_path):
    """重命名指定文件夹中的所有图片文件"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file)
    
    # 按文件名排序，确保重命名的一致性
    image_files.sort()
    
    if not image_files:
        print(f"文件夹中没有找到图片文件: {folder_path}")
        return
    
    print(f"处理文件夹: {folder_path}")
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 创建临时文件夹避免命名冲突
    temp_folder = os.path.join(folder_path, "temp_rename")
    os.makedirs(temp_folder, exist_ok=True)
    
    try:
        # 先移动到临时文件夹
        for i, old_filename in enumerate(image_files):
            old_path = os.path.join(folder_path, old_filename)
            temp_path = os.path.join(temp_folder, f"temp_{i}_{old_filename}")
            shutil.move(old_path, temp_path)
        
        # 再从临时文件夹移动回来并重命名
        temp_files = os.listdir(temp_folder)
        temp_files.sort()
        
        for i, temp_filename in enumerate(temp_files):
            temp_path = os.path.join(temp_folder, temp_filename)
            # 保持原有扩展名
            original_name = temp_filename.split("_", 2)[-1]  # 去掉temp_i_前缀
            file_ext = os.path.splitext(original_name)[1]
            new_filename = f"img{i+1}{file_ext}"
            new_path = os.path.join(folder_path, new_filename)
            
            shutil.move(temp_path, new_path)
            print(f"  {original_name} -> {new_filename}")
        
        # 删除临时文件夹
        os.rmdir(temp_folder)
        
    except Exception as e:
        print(f"重命名过程中出错: {e}")
        # 如果出错，尝试恢复文件
        if os.path.exists(temp_folder):
            for temp_file in os.listdir(temp_folder):
                temp_path = os.path.join(temp_folder, temp_file)
                original_name = temp_file.split("_", 2)[-1]
                original_path = os.path.join(folder_path, original_name)
                shutil.move(temp_path, original_path)
            os.rmdir(temp_folder)

def main():
    """主函数：遍历所有类别文件夹并重命名图片"""
    data_root = "data"
    
    if not os.path.exists(data_root):
        print(f"数据文件夹不存在: {data_root}")
        return
    
    # 类别列表
    categories = ["加载不全", "弹窗", "桌面页", "登录页", "空白页"]
    
    # 处理训练集和验证集
    for split in ["train", "val"]:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"文件夹不存在: {split_path}")
            continue
            
        print(f"\n处理 {split} 数据集:")
        print("=" * 50)
        
        for category in categories:
            category_path = os.path.join(split_path, category)
            rename_images_in_folder(category_path)
    
    print("\n重命名完成！")

if __name__ == "__main__":
    main()
