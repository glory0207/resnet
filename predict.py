import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import create_model

class MobileScreenPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
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
    
    def predict_single(self, image_path):
        """预测单张图片"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': self.classes[predicted_class],
            'confidence': confidence,
            'all_probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
    
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

def main():
    # 使用示例
    predictor = MobileScreenPredictor('mobile_screen_classifier.pth')
    
    # 预测单张图片
    image_path = 'test_image.jpg'  # 替换为你的图片路径
    if os.path.exists(image_path):
        result = predictor.predict_single(image_path)
        print(f"预测结果: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("所有类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    
    # 预测文件夹中的所有图片
    test_folder = 'test_images'  # 替换为你的测试图片文件夹路径
    if os.path.exists(test_folder):
        results = predictor.predict_folder(test_folder)
        print(f"\n文件夹预测结果:")
        for result in results:
            if 'error' not in result:
                print(f"{os.path.basename(result['image_path'])}: {result['predicted_class']} (置信度: {result['confidence']:.4f})")
            else:
                print(f"{os.path.basename(result['image_path'])}: 处理失败 - {result['error']}")

if __name__ == '__main__':
    main()