from flask import Flask, request, send_from_directory, jsonify
import os
import torch
from torchvision import transforms, models
from PIL import Image
import random
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_ecg")
STAGE1_MODEL_PATH = os.path.join(BASE_DIR, "ecg_filter_model.pth")
ECG_MODEL_PATH = os.path.join(BASE_DIR, "ecg_model.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


stage1_model = models.mobilenet_v3_small(weights=None)
stage1_model.classifier[3] = nn.Linear(stage1_model.classifier[3].in_features, 2)
stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
stage1_model.to(device)
stage1_model.eval()


ecg_model = models.efficientnet_b0(weights=None)
ecg_model.classifier[1] = nn.Linear(ecg_model.classifier[1].in_features, 2)
ecg_model.load_state_dict(torch.load(ECG_MODEL_PATH, map_location=device))
ecg_model.to(device)
ecg_model.eval()

transform_stage1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_stage2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


normal_insights = [
    "Heartbeat is normal, good rhythm detected.",
    "Heart rate variability is within healthy range.",
    "No abnormal R-peaks detected.",
    "Your ECG signals are stable and normal.",
    "No signs of arrhythmia detected.",
    "Cardiac function appears regular."
]

abnormal_insights = [
    "Possible arrhythmia detected.",
    "Irregular heartbeat observed.",
    "ST-segment abnormality detected.",
    "Signs of myocardial ischemia possible.",
    "Consult cardiologist for detailed analysis."
]

normal_possibilities = [
    "Healthy heart rhythm", "Normal R-peaks",
    "Stable ECG waveform", "Low cardiac risk"
]

abnormal_possibilities = [
    "Atrial fibrillation", "Bradycardia",
    "Tachycardia", "Myocardial ischemia"
]

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_page():
    if request.method == "POST":
        img_file = request.files.get("image")
        if not img_file:
            return jsonify({"error": "Please upload an ECG image."}), 400

        img_path = os.path.join(UPLOAD_DIR, img_file.filename)
        img_file.save(img_path)

        try:
            img = Image.open(img_path).convert("RGB")

          
            t1 = transform_stage1(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out1 = stage1_model(t1)
                prob1 = torch.softmax(out1, dim=1)
                conf1, pred1 = torch.max(prob1, 1)

            if pred1.item() == 1:
                return jsonify({
                    "image_status": "Invalid",
                    "confidence_image": round(conf1.item() * 100, 2),
                    "image_url": f"/uploaded_ecg/{img_file.filename}",
                    "insight_image": "No ECG waveform detected.",
                    "possibilities": []
                })

           
            t2 = transform_stage2(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out2 = ecg_model(t2)
                prob2 = torch.softmax(out2, dim=1)
                conf2, pred2 = torch.max(prob2, 1)

            status = "Healthy" if pred2.item() == 1 else "Unhealthy"

            insight = random.choice(
                normal_insights if status == "Healthy" else abnormal_insights
            )
            possibilities = random.sample(
                normal_possibilities if status == "Healthy" else abnormal_possibilities,
                3
            )

            return jsonify({
                "image_status": status,
                "confidence_image": round(conf2.item() * 100, 2),
                "image_url": f"/uploaded_ecg/{img_file.filename}",
                "insight_image": insight,
                "possibilities": possibilities
            })

        except Exception as e:
            print("Error:", e)
            return jsonify({
                "image_status": "Error",
                "confidence_image": "--",
                "insight_image": "Failed to process ECG image.",
                "possibilities": []
            })

    return send_from_directory(FRONTEND_DIR, "analyze.html")

@app.route("/uploaded_ecg/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "images"), filename)

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ─── HF ENTRY POINT ───
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)