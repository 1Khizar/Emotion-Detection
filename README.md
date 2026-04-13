<div align="center">
  <h1>📷 FaceVibe — AI Emotion Detection</h1>
  <p><strong>The easiest way to analyze facial expressions.</strong></p>
  <p>Powered by a custom <b>PyTorch CNN</b> backend and a modern <b>Next.js 16</b> frontend interface.</p>
</div>

<br />

## ✨ Key Features

- **📷 Web Camera Analysis**: Take a live picture using your computer camera and analyze emotions instantly.
- **📁 Photo Upload**: Choose a picture from your computer via file upload to see what emotions it reveals.
- **⚡ Real-Time Processing**: The backend efficiently extracts facial regions and runs them through a lightweight Convolutional Neural Network (CNN) for rapid inference.
- **🎨 Clean & Modern UI**: Built with a spacious, polished, and lightweight user experience.

---

## 🛠 Tech Stack

### Frontend
- **Framework:** Next.js 16.2, React 19
- **Language:** TypeScript
- **Styling:** Inline CSS and Tailwind CSS (PostCSS)
- **Features:** Client-side interactions (`"use client"`), fluid hover animations.

### Backend
- **Server:** FastAPI
- **AI/ML:** PyTorch, TorchVision
- **Vision:** OpenCV
- **Data:** NumPy, Pydantic

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**

### 1. Backend Setup

Open a terminal in the project root (`Emotion Detection/`):

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install all necessary backend dependencies 
pip install -r requirements.txt

# 3. Navigate to the backend directory
cd backend

# 4. Run the API Server
python main.py
```
> **Troubleshooting Missing Modules**: If you get a `ModuleNotFoundError` (like missing `torch`), make sure your `venv` is fully activated and re-run the `pip install -r requirements.txt` command. The requirements file has been optimized to resolve strict version conflicts on Windows.

The backend API will run at `http://localhost:8000`.

### 2. Frontend Setup

Open a **new** terminal window and keep the backend running at the same time.

```powershell
# 1. Navigate to the frontend directory
cd frontend

# 2. Install NPM dependencies
npm install

# 3. Run the development server
npm run dev
```

The FaceVibe application will now be accessible at `http://localhost:3000`. 

---

## 🧠 Model Details

FaceVibe is powered by a custom deep learning Convolutional Neural Network:
- **Input:** 48x48 Grayscale pixels.
- **Detected Labels:** `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`.

The trained model weights are stored securely in `emotion_model.pth` located at the root of the project.

---

<div align="center">
  <p><i>Building intuitive tools to analyze human emotion.</i></p>
</div>

---

## 👨‍💻 Author

**Khizar Ishtiaq**  
*AI Engineer*  
- **LinkedIn:** [www.linkedin.com/in/khizar-ishtiaq-](https://www.linkedin.com/in/khizar-ishtiaq-)
- **Portfolio:** [khizarai.vercel.app](https://khizarai.vercel.app)
