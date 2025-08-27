import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import medmnist
from medmnist import INFO, Evaluator
import os

# 從專案根目錄匯入模型
from src.model import SimpleCNN

def train(args):
    data_flag = args.data_flag
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # 資料預處理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5] * n_channels, std=[.5] * n_channels)
    ])
    
    # 載入 MedMNIST 資料集，已將 'dataset' 修正為 'python_class'
    train_dataset = getattr(medmnist, info['python_class'])(split='train', transform=data_transform, download=download)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 選擇設備 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes).to(device)
    
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 訓練迴圈
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
            else:
                labels = labels.squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
    # 儲存模型權重
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))
    print("模型訓練完成並已儲存！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedMNIST 訓練腳本")
    parser.add_argument('--data_flag', type=str, default='pathmnist', help="MedMNIST 資料集名稱")
    parser.add_argument('--epochs', type=int, default=5, help="訓練週期")
    parser.add_argument('--batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="學習率")
    args = parser.parse_args()
    
    train(args)