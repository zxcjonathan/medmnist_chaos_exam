import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src.model import SimpleCNN
from src.utils import get_gradcam_heatmap

# 載入模型
@st.cache(allow_output_mutation=True)
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 影像預處理（包含直方圖均衡化）
def preprocess_for_app(image_pil, size=(224, 224)):
    img_gray = image_pil.convert('L')
    img_eq = ImageOps.equalize(img_gray)
    img_rgb = img_eq.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_rgb, transform(img_rgb).unsqueeze(0)

# 顯示帶有熱力圖的影像
def show_heatmap(image_pil, heatmap):
    heatmap = cv2.resize(heatmap, (image_pil.width, image_pil.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(np.array(image_pil.convert('RGB')), 0.6, heatmap, 0.4, 0)
    superimposed_img_pil = Image.fromarray(superimposed_img)
    st.image(superimposed_img_pil, caption='模型關注的區域 (熱力圖)', use_column_width=True)

# Streamlit 介面
st.title("胸部X光片肺炎輔助篩查工具")
st.write("請上傳一張胸部X光片影像，AI將為您進行輔助診斷。")

uploaded_file = st.file_uploader("選擇一張影像...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上傳影像', use_column_width=True)
    st.write("")
    
    # 執行預處理
    display_image, processed_tensor = preprocess_for_app(image)
    
    with st.spinner('正在分析中...'):
        with torch.no_grad():
            output = model(processed_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prediction = torch.argmax(probabilities).item()

    # 結果展示
    class_names = ["正常", "肺炎"]
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction].item()

    st.write(f"### AI診斷結果：{predicted_class}")
    st.write(f"### 置信度：{confidence:.2%}")
    
    if predicted_class == "肺炎":
        st.error("警告：模型診斷為肺炎。請務必諮詢專業醫療人員！")
    else:
        st.success("模型診斷為正常。")
        
    # 顯示 Grad-CAM 熱力圖
    st.markdown("---")
    st.subheader("模型判斷依據 (可解釋性 AI)")
    heatmap = get_gradcam_heatmap(model, processed_tensor, class_idx=prediction)
    show_heatmap(image, heatmap)