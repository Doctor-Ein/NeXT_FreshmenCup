from transformers import AutoTokenizer, AutoModelForImageClassification
from PIL import Image
from torchvision import transforms
import torch

# 设置模型路径
model_name = "./local_models/VisCoT-13b-336"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 加载和预处理图像
image_path = "./Data/Paper/MinerU_Res/AlexNet/images/ae801d13c783d10fcc5c9b8b9198e4ebe0ead3e7fbd040082f2248274f12c7fc.jpg"
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)  # 增加批次维度

# 使用模型生成摘要
inputs = tokenizer(images=input_tensor, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("生成的摘要：", generated_text)
