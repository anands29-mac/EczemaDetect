# 🩺 EczemaDetect  
### AI-Powered Eczema Detection & Severity Classification  
Built with **FastAPI**, **React**, **Three.js**, **MongoDB**, and a **CNN-based deep learning model**

---

## 📌 Overview  
**EczemaDetect** is an end-to-end AI system that analyzes skin images to detect eczema and classify its type and severity.  
It combines **deep learning**, **3D visualization**, and full-stack engineering to create a powerful dermatology-focused diagnostic assistant.

This project uses:
- 🧠 **CNN model** for image classification  
- ⚡ **FastAPI** backend for ML inference + API  
- 🗄️ **MongoDB** for storing survey responses, predictions, and images  
- 💻 **React** as the main frontend  
- 🌐 **Three.js** for interactive 3D visualizations (severity heatmaps, skin models)  

---

## ✨ Features

### 🔍 1. **AI-based Eczema Detection**
- Classifies **6 types** of eczema:
  - Atopic  
  - Contact  
  - Dyshidrotic  
  - Nummular  
  - Seborrheic  
  - Stasis  
- CNN trained on curated dermatology datasets  
- Outputs **type**, **severity score**, and **confidence levels**

---

### 📝 2. **Symptom Survey Module**
A questionnaire that collects:
- Dryness  
- Redness levels  
- Itching intensity  
- Flare-up frequency  
- Family/medical history  

Survey + image prediction = a **more complete diagnosis**, stored securely in MongoDB.

---

### 🖼️ 3. **Image Upload & Prediction**
Users can upload a close-up skin image.  
FastAPI returns:
- Predicted eczema type  
- Severity score  
- Model confidence  
- Heatmap overlay (optional future feature)

---

### 🧊 4. **3D Visualizations (Three.js)**
Adds “wow factor” with:
- 3D rotating skin model  
- Severity heatmap overlays  
- Interactive highlighting of affected regions  
- Smooth animations using Three.js or React-Three-Fiber  

---

### 📊 5. **User Dashboard**
Displays:
- Prediction history  
- Symptom trends  
- 3D visualization  
- Severity progress over time  
- Exportable PDF report (future)

---

## 🏗️ Tech Stack

### **Frontend**
- React.js  
- Three.js / React-Three-Fiber  
- Axios  

### **Backend**
- FastAPI  
- Uvicorn  
- Pydantic  
- TensorFlow / Keras (CNN model)

### **Database**
- MongoDB (Atlas)

### **Misc**
- Python  
- Node.js  
- Git/GitHub  
- Docker (optional deployment)

---

## 📁 Project Structure

```bash
EczemaDetect/
│
├── backend/
│   ├── app.py                # FastAPI entry point
│   ├── models/
│   │   └── eczema_cnn.h5     # Trained ML model
│   ├── routes/
│   │   ├── survey.py
│   │   ├── predict.py
│   │   └── results.py
│   ├── utils/
│       ├── preprocessing.py
│       └── model_loader.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── threejs/          # 3D models + scenes
│   │   └── App.js
│   └── package.json
│
└── README.md
