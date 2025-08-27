import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from PIL import Image, ImageOps
from torchvision import transforms
import cv2

def calculate_metrics(y_true, y_pred, average='weighted'):
    # ... (保持不變) ...

def plot_confusion_matrix(y_true, y_pred, class_names):
    # ... (保持不變) ...

# 2. 影像前處理的擴展：直方圖均衡化
def preprocess_image(image_path, size=(224, 224)):
    """
    載入並預處理影像，包含直方圖均衡化。
    """
    img = Image.open(image_path).convert('L') # 轉為灰階
    
    # 直方圖均衡化
    img = ImageOps.equalize(img)

    # 轉為 RGB 格式以適應預訓練模型
    img = img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# 3. 可解釋性 AI：Grad-CAM 實現
class GradCAM:
    def __init__(self, model, target_layer_name="backbone.layer4"):
        self.model = model
        self.target_layer = self.find_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        self.register_hooks()
    
    def find_target_layer(self, layer_name):
        # 尋找目標卷積層
        module_list = [name for name, _ in self.model.named_modules()]
        for module_name in module_list:
            if module_name == layer_name:
                return getattr(self.model, layer_name)
        raise ValueError(f"Target layer '{layer_name}' not found.")

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        target = output[0, class_idx]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.size(0)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        heatmap /= np.max(heatmap)
        
        return heatmap

def get_gradcam_heatmap(model, img_tensor, class_idx=None):
    """
    生成並返回 Grad-CAM 熱力圖。
    """
    grad_cam = GradCAM(model)
    heatmap = grad_cam(img_tensor, class_idx)
    return heatmap