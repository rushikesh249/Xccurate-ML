# 🧠 Xccurate-ML: Brain Tumor Detection System

---

**Team Name:** Team INNOVISIONERS  \
**Team Members:** Aditya Chavan, Rushikesh Ambhore, Atharva Agey, Pranav Dawange  \
**Project Name:** Xccurate-ML  

> Xccurate-ML is an advanced AI-powered diagnostic tool that leverages deep learning to analyze MRI brain scans for the detection and classification of brain tumors. Designed for both healthcare professionals and non-experts, it delivers fast, accurate, and accessible results, supporting early diagnosis and improved patient outcomes.

**Tech Stack:**
- **Backend:** Python 3.9+, Flask, TensorFlow 2.x, Keras, NumPy, Pillow
- **Frontend:** React (Vite), TypeScript, shadcn/ui, TailwindCSS
- **Other:** Flask-CORS, REST API

**Dataset Used:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

---

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow 2.x"/>
  <img src="https://img.shields.io/badge/Keras-2.x-red.svg" alt="Keras"/>
  <img src="https://img.shields.io/badge/React-18.x-61dafb.svg" alt="React"/>
</p>

> **AI-powered MRI analysis for fast, accurate, and accessible brain tumor detection.**

---

## 📚 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [How it Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Backend API Usage](#backend-api-usage)
- [Frontend Workflow](#frontend-workflow)
- [Model Architecture](#model-architecture)
- [Dataset Information](#dataset-information)
- [Model Training Details](#model-training-details)
- [Results & Insights](#results--insights)
- [Future Scope](#future-scope)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## 🚀 Project Overview

An advanced **AI + ML diagnostic system** that analyzes **MRI brain scans** to detect the presence and type of brain tumor with high accuracy. Designed for both healthcare professionals and non-experts, this tool simplifies medical scan interpretation and supports early diagnosis.

### 🎯 Core Objective
Empower rapid, accessible tumor screening for all—reducing diagnosis time, aiding remote areas, and assisting clinicians with instant, reliable insights.

---

## 🧩 Key Features
- 🩺 **Brain Tumor Classification** — Detects and classifies:
  - *Pituitary Tumor*
  - *Glioma Tumor*
  - *Meningioma Tumor*
  - *No Tumor (Healthy)*
- ⚙️ **Deep Learning Model:** Transfer learning with **ResNet50V2**
- 🧠 **Input:** MRI scan (JPG/PNG)
- 📊 **Output:** Tumor prediction and type in human-readable format
- 💻 **Cross-platform Ready:** Integrate into mobile/web apps
- 🔒 **Locally Secure:** No cloud upload—model runs locally for privacy
- 🌐 **Modern UI:** Fast, responsive, and accessible web interface

---

## 🛠️ How it Works

1. **Upload MRI Image** → 2. **Image Preprocessing** → 3. **Model Prediction (Backend API)** → 4. **Result Interpretation (Frontend UI)**

---

## 📁 Project Structure

```text
.
├── backend/
│   ├── main.py                # Flask API for model inference
│   ├── best_model_augmented.keras # Trained deep learning model
│   ├── requirements.txt       # Backend dependencies
│   └── Testing/               # Test MRI images (by class)
├── src/
│   ├── App.tsx                # React app entry
│   ├── pages/                 # Main pages (Index, ScanPage, NotFound)
│   ├── components/            # UI components (Hero, HowItWorks, etc.)
│   └── ...                    # Styles, assets, hooks, etc.
├── package.json               # Frontend dependencies
├── README.md                  # Project documentation
└── ...
```

---

## ⚡ Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- pip, npm/yarn

### Backend Setup
1. Navigate to `backend/`:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:5000`.

### Frontend Setup
1. Navigate to the project root:
   ```bash
   cd .. # if in backend/
   npm install
   npm run dev
   ```
2. The React app will be available at `http://localhost:5173` (default Vite port).

---

## 🔌 Backend API Usage

### Endpoints
- `GET /status` — Check model and server health
  - **Response:** `{ status: "ok", model_loaded: true, input_shape: [height, width] }`
- `POST /predict` — Predict tumor type from uploaded image
  - **Request:** Multipart form-data with key `file` (image file)
  - **Response:**
    ```json
    {
      "predicted_class": "glioma",
      "confidence": 0.97,
      "all_class_probabilities": {
        "glioma": 0.97,
        "meningioma": 0.01,
        "notumor": 0.01,
        "pituitary": 0.01
      }
    }
    ```

---

## 🖥️ Frontend Workflow

- **Landing Page:** Explains the product, features, and workflow.
- **Scan Page:**
  - User uploads an MRI/medical image (JPG/PNG).
  - The app checks backend status (`/status`).
  - On upload, the image is sent to the backend (`/predict`).
  - Results (diagnosis, confidence, detailed findings, recommendations) are displayed in a modern, responsive UI.
  - Medical disclaimer and guidance are provided.
- **Error Handling:** User-friendly error messages for backend/API issues.
- **Routing:** `/` (home), `/scan` (analyze), `*` (404 Not Found)

---

## 🧠 Model Architecture

- **Base Model:** ResNet50V2 (Pre-trained on ImageNet)
- **Approach:** Transfer Learning
- **Input Shape:** `150x150x3`
- **Output Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)

---

## 📊 Dataset Information

- **Source:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes:** `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`
- **Training Samples:** 5,712
- **Validation Samples:** 1,311
- **Test Samples:** 1,311
- **Image Size:** 150×150 pixels (normalized RGB)

---

## 🧪 Model Training Details

| Parameter          | Description                                  |
| ------------------ | -------------------------------------------- |
| Epochs             | 25                                           |
| Batch Size         | 32                                           |
| Image Augmentation | Rotation, Zoom, Flip, Brightness Adjustments |
| Callbacks          | EarlyStopping, ModelCheckpoint               |
| GPU Used           | NVIDIA Tesla T4 (Google Colab)               |
| Training Time      | ~100 mins                                    |
| Final Accuracy     | 96.8% (Validation), 97.4% (Test)             |

---

## 📈 Results & Insights

- **Training Accuracy:** 97.8%
- **Validation Accuracy:** 96.8%
- **Loss:** 0.09
- **Observation:** Model generalizes well and correctly differentiates tumor regions.
- **Confusion Matrix:** High precision on all tumor types.

---

## 🔮 Future Scope

- Integrate **CT, PET, and Ultrasound** image classification
- Add **explainability (Grad-CAM)** to highlight tumor regions
- Build **interactive dashboard** for visual insights
- Deploy model as **API microservice** for hospitals
- Integrate **voice-based report summarizer** for accessibility

---

## 🧑‍💻 Contributors

**Team INNOVISIONERS**
- Aditya Chavan — Machine Learning Engineer
- Rushikesh Ambhore — UI/UX & Frontend Designer
- Atharva Agey — Backend Developer
- Pranav Dawange — Data Scientist

---

## 📜 License

This project is released under the **MIT License**. You are free to use, modify, and distribute this work with proper attribution.

---

## 🙏 Acknowledgements

Special thanks to:
- TensorFlow & Keras teams for powerful open-source libraries
- Kaggle dataset contributors
- Hackathon mentors and reviewers for their guidance

---

## 📬 Contact

For questions, suggestions, or collaborations:
- **Email:** [aditya.chavan24@vit.edu](mailto:aditya.chavan24@vit.edu)
- **LinkedIn:** [Aditya Chavan](https://www.linkedin.com/in/aadii-chavan)

---

> 💡 *“AI will not replace doctors, but doctors who use AI will replace those who don’t.”*
