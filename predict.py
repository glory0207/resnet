import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import create_model

class MobileScreenPredictor:
    def __init__(self, model_path, device='cuda', confidence_threshold=0.6):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        
        self.model = create_model(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"模型已加载，异常类别: {self.classes}")
        print(f"置信度阈值: {confidence_threshold} (低于此值判断为正常页面)")
    
    def predict_single(self, image_path):
        """预测单张图片，支持正常页面判断"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probabilities, dim=1)
            
            max_confidence = max_prob.item()
            predicted_class = self.classes[predicted_idx.item()]
            
            # 判断是否为正常页面
            is_normal = max_confidence < self.confidence_threshold
            
            result = {
                'predicted_class': "正常页面" if is_normal else predicted_class,
                'confidence': (1.0 - max_confidence) if is_normal else max_confidence,
                'is_normal': is_normal,
                'abnormal_type': predicted_class if not is_normal else None,
                'max_abnormal_confidence': max_confidence,
                'all_probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            }
            
            return result
    
    def predict_batch(self, image_paths):
        """批量预测多张图片"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def predict_folder(self, folder_path):
        """预测文件夹中的所有图片"""
        image_paths = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
        
        return self.predict_batch(image_paths)
    
    def set_confidence_threshold(self, threshold):
        """调整置信度阈值"""
        self.confidence_threshold = threshold
        print(f"置信度阈值已调整为: {threshold}")
    
    def analyze_predictions(self, results):
        """分析预测结果的统计信息"""
        normal_count = sum(1 for r in results if r.get('is_normal', False))
        abnormal_count = len(results) - normal_count
        
        abnormal_types = {}
        for result in results:
            if not result.get('is_normal', True) and 'abnormal_type' in result:
                abnormal_type = result['abnormal_type']
                abnormal_types[abnormal_type] = abnormal_types.get(abnormal_type, 0) + 1
        
        print(f"\n=== 预测结果统计 ===")
        print(f"总计图片: {len(results)}")
        print(f"正常页面: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        print(f"异常页面: {abnormal_count} ({abnormal_count/len(results)*100:.1f}%)")
        
        if abnormal_types:
            print(f"\n异常类型分布:")
            for abnormal_type, count in abnormal_types.items():
                print(f"  {abnormal_type}: {count} ({count/abnormal_count*100:.1f}%)")
        
        return {
            'total': len(results),
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'abnormal_types': abnormal_types
        }

def main():
    # 使用示例
    predictor = MobileScreenPredictor('mobile_screen_classifier.pth', confidence_threshold=0.5)
    
    # 预测单张图片
    image_path = 'test_image.jpg'  # 替换为你的图片路径
    if os.path.exists(image_path):
        result = predictor.predict_single(image_path)
        print(f"\n=== 预测结果 ===")
        print(f"图片: {image_path}")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"是否正常页面: {'是' if result['is_normal'] else '否'}")
        if not result['is_normal']:
            print(f"异常类型: {result['abnormal_type']}")
        
        print(f"\n所有异常类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    
    # 预测文件夹中的所有图片
    test_folder = 'tanchuang_images'  # 测试空白页类别
    if os.path.exists(test_folder):
        results = predictor.predict_folder(test_folder)
        print(f"\n=== 文件夹预测结果 ({test_folder}) ===")
        
        normal_count = sum(1 for r in results if r.get('is_normal', False))
        abnormal_count = len(results) - normal_count
        
        print(f"测试了 {len(results)} 张图片:")
        print(f"  判断为正常页面: {normal_count} 张")
        print(f"  判断为异常页面: {abnormal_count} 张")
        
        print(f"\n详细结果:")
        for result in results:
            if 'error' not in result:
                status = "正常页面" if result['is_normal'] else f"{result['abnormal_type']}"
                print(f"  {os.path.basename(result['image_path'])}: {status} (置信度: {result['confidence']:.3f})")
            else:
                print(f"  {os.path.basename(result['image_path'])}: 处理失败 - {result['error']}")
    
    # 展示如何调整置信度阈值
    print(f"\n=== 调整置信度阈值示例 ===")
    predictor.set_confidence_threshold(0.5)  # 更严格的阈值，更容易判断为正常页面
    
    # 如果需要统计分析
    if test_folder and os.path.exists(test_folder):
        results = predictor.predict_folder(test_folder)
        stats = predictor.analyze_predictions(results)
        print(f"\n统计分析完成，异常类型数: {len(stats['abnormal_types'])}")

if __name__ == '__main__':
    main()