# app.py
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model_def import MiniVGG

app = Flask(__name__)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniVGG(num_classes=2)
model.load_state_dict(torch.load("model/mini_vgg_model.pth", map_location=device))
model.to(device)
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 输入尺寸要与你训练时一致
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
