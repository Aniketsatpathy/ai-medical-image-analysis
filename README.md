# 🩺 AI-Powered Medical Image Analysis System

## 📌 Overview
This project is an AI-based medical image analysis system that detects pneumonia from chest X-ray images using deep learning and computer vision techniques.

It provides:
- Automated disease detection using a trained deep learning model
- Visual explanations using Grad-CAM (Explainable AI)
- Interactive web interface using Streamlit for real-time predictions

---

## 🚨 Problem Statement
Early detection of pneumonia is critical, but traditional diagnosis:
- Requires expert radiologists
- Takes time
- Can be prone to human error

This system assists in faster, scalable, and reliable diagnosis using AI.

---

## 🏥 Industry Relevance
This solution is applicable in:
- Hospitals
- Diagnostic laboratories
- Radiology centers
- Health-tech startups

### Key Benefits:
- Faster diagnosis
- Reduced workload on doctors
- AI-assisted decision-making
- Scalable screening system

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Data Processing:** NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Web Interface:** Streamlit  

---

## 📊 Dataset
- Public Chest X-ray dataset  
- Classes:
  - Normal
  - Pneumonia  

> Note: Dataset is not included due to size constraints.
Here is the Download Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
---

## ⚙️ System Architecture

1. Image Input (X-ray)
2. Preprocessing (resize, normalization)
3. Feature Extraction (MobileNetV2)
4. Classification Layer
5. Prediction Output
6. Grad-CAM Visualization

---

## 📈 Model Performance

- **Test Accuracy:** ~87%  
- **Pneumonia Recall:** ~97%  
- **Precision (Pneumonia):** ~84%  

### Confusion Matrix:
[[162 72]
[ 10 380]]


👉 The model is highly effective at detecting pneumonia cases, minimizing missed diagnoses.

---

## 🔍 Grad-CAM (Explainable AI)

Grad-CAM highlights regions in the X-ray where the model is focusing.

This:
- Increases trust in AI predictions
- Provides visual justification
- Helps doctors interpret results

---

## 🖼️ Screenshots

### 🔹 Streamlit UI
![UI](assets/ui.png)

### 🔹 Prediction Output
![Prediction](assets/prediction.png)

### 🔹 Grad-CAM Heatmap
![GradCAM](assets/gradcam.png)

### 🔹 Training Graph
![Training](assets/training.png)

### 🔹 Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

---

## 💻 Installation

```
git clone https://github.com/Aniketsatpathy/ai-medical-image-analysis.git
cd ai-medical-image-analysis

python -m venv myenv
.\myenv\Scripts\activate

pip install -r requirements.txt



▶️ Usage

Run Main Pipeline
python main.py

Run Streamlit UI
streamlit run app.py



📁 Project Structure
AI-Medical-Image-Analysis/
│
├── src/
├── data/
├── models/
├── outputs/
├── assets/
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore



🎯 Learning Outcomes

Through this project, I gained hands-on experience in:

Building end-to-end ML pipelines
Transfer learning using MobileNetV2
Debugging real-world ML systems
Implementing explainable AI (Grad-CAM)
Developing interactive ML applications using Streamlit
Structuring projects for production readiness



🚀 Future Improvements
Deploy application on cloud
Support multi-disease detection
Improve dataset balancing
Add REST API integration
Enhance UI/UX

👨‍💻 Author
Aniket Satpathy

⭐ If you found this useful, consider giving a star!
