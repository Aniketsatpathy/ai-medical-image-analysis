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
> Here is the Download Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

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
![Confusion Matrix](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/outputs/confusion_matrix.png)

👉 The model is highly effective at detecting pneumonia cases, minimizing missed diagnoses.

---

## 🔍 Grad-CAM (Explainable AI)

Grad-CAM highlights regions in the X-ray where the model is focusing.

This:
- Increases trust in AI predictions
- Provides visual justification
- Helps doctors interpret results

### Grad-CAM Visualization Example:
![Grad-CAM Result](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/outputs/gradcam_result.jpg)

---

## 🖼️ Screenshots & Visualizations

### 🔹 Interactive Dashboard UI
![Dashboard UI](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Dashboard_UI.png)

### 🔹 Dashboard Heatmap Analysis
![Dashboard Heatmap](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Dashboard_HeatMap.png)

### 🔹 Dashboard Closeup - Heatmap Details
![Dashboard Closeup](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Dashboard_Closeup_heatmap.png)

### 🔹 Input Radiology Image
![Input Radiology](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Input_Radiology.png)

### 🔹 Diagnosis Output
![Diagnosis](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Diagnosis.png)

### 🔹 Heatmap Visualization
![Heatmap](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Heatmap.png)

### 🔹 Analysis Report
![Analysis Report](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Analysis%20Report.png)

### 🔹 Training Performance Graph
![Training Graph](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/outputs/training.png)

### 🔹 Loss Curve During Training
![Loss Curve](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/outputs/loss.png)

### 🎥 Dashboard Demo Video
[Watch Dashboard Demo](https://github.com/Aniketsatpathy/ai-medical-image-analysis/raw/main/assets/Dashboard_Demo.mp4)

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/Aniketsatpathy/ai-medical-image-analysis.git
cd ai-medical-image-analysis

# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
.\myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Main Pipeline
```bash
python main.py
```

### Run Streamlit Web Interface
```bash
streamlit run app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
AI-Medical-Image-Analysis/
│
├── src/                      # Source code modules
├── data/                      # Dataset directory
├── models/                    # Trained model files
├── outputs/                   # Generated outputs (graphs, matrices, predictions)
│   ├── confusion_matrix.png
│   ├── gradcam_result.jpg
│   ├── loss.png
│   └── training.png
├── assets/                    # UI screenshots and demo files
│   ├── Dashboard_UI.png
│   ├── Dashboard_HeatMap.png
│   ├── Dashboard_Closeup_heatmap.png
│   ├── Input_Radiology.png
│   ├── Diagnosis.png
│   ├── Heatmap.png
│   ├── Analysis Report.png
│   ├── gradcam_result.jpg
│   └── Dashboard_Demo.mp4
├── app.py                     # Streamlit application
├── main.py                    # Main training pipeline
├── requirements.txt           # Project dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore file
```

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience in:

- Building end-to-end ML pipelines
- Transfer learning using MobileNetV2
- Debugging real-world ML systems
- Implementing explainable AI (Grad-CAM)
- Developing interactive ML applications using Streamlit
- Structuring projects for production readiness

---

## 🚀 Future Improvements

- [ ] Deploy application on cloud (AWS, Azure, GCP)
- [ ] Support multi-disease detection
- [ ] Improve dataset balancing with data augmentation
- [ ] Add REST API integration
- [ ] Enhance UI/UX with additional features
- [ ] Implement model versioning and tracking
- [ ] Add real-time monitoring and logging

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Aniket Satpathy**

GitHub: [@Aniketsatpathy](https://github.com/Aniketsatpathy)

⭐ If you found this useful, consider giving a star!

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

---

## 📞 Contact & Support
satpathyaniket81@gmail.com
For questions or collaboration inquiries, feel free to reach out!
