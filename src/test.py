import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import argparse
import medmnist
from medmnist import INFO
import os

from src.model import SimpleCNN

def evaluate(args):
    data_flag = args.data_flag
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5] * n_channels, std=[.5] * n_channels)
    ])
    
    # 載入 MedMNIST 測試資料集
    test_dataset = getattr(medmnist, info['python_class'])(split='test', transform=data_transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes).to(device)
    
    # 載入模型權重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # 取得預測的類別
            if task == 'multi-class':
                _, predicted = torch.max(outputs.data, 1)
            else: # 對於二元分類，需要閾值
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 將列表轉換為 NumPy 陣列
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).squeeze()

    # 使用 scikit-learn 計算所有評估指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("--- 模型評估結果 ---")
    print(f"準確率 (Accuracy): {accuracy:.4f}")
    print(f"加權精確率 (Precision): {precision:.4f}")
    print(f"加權召回率 (Recall): {recall:.4f}")
    print(f"加權 F1-Score: {f1:.4f}")
    print("\n混淆矩陣 (Confusion Matrix):")
    print(cm)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedMNIST 測試腳本")
    parser.add_argument('--data_flag', type=str, default='pathmnist', help="MedMNIST 資料集名稱")
    parser.add_argument('--batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help="模型權重路徑")
    args = parser.parse_args()
    
    evaluate(args)