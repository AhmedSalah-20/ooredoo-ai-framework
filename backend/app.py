# backend/app.py - FIXED VERSION

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path
import yaml

# ✅ أضف المسار الصحيح
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ الآن يمكنك استيراد من src
try:
    from src.unified_pipeline import UnifiedPipeline
    from src.trainers.llm_trainer import LLMTrainer
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ============= INIT =============
app = FastAPI(
    title="Ooredoo AI Framework",
    description="STT + LLM + TTS API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# ✅ FIX: Load config with UTF-8 encoding
try:
    with open("configs/models/llm_config.yaml", encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    print(f"✅ Config loaded")
except FileNotFoundError:
    print("⚠️ Config file not found, using defaults")
    llm_config = {
        "model": {"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        "training": {"output_dir": "./models/tunisian-lm-final"}
    }
except Exception as e:
    print(f"⚠️ Config error: {e}")
    llm_config = {}

# Global state
trainer = None
try:
    trainer = LLMTrainer(llm_config)
    trainer.load_model()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Model load warning: {e}")
    trainer = None

# ============= ROUTES =============

@app.get("/")
async def root():
    """Serve main HTML page"""
    return FileResponse("frontend/static/index.html")

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "models_ready": trainer is not None,
        "gpu_available": True
    }

# ============= LLM ENDPOINTS =============

@app.post("/api/llm/generate")
async def generate_text(text: str, max_length: int = 100):
    """
    LLM Inference: Text → Text
    """
    try:
        if trainer is None:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        # Real inference
        response = trainer.inference(text)
        
        return {
            "status": "success",
            "input": text,
            "output": response
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============= STT ENDPOINTS (Mock for now) =============

@app.post("/api/stt/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    STT: Audio → Text
    """
    try:
        # TODO: Implement real STT when ready
        mock_text = "أهلا وسهلا في منصة Ooredoo"
        
        return {
            "status": "success",
            "transcription": mock_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============= TTS ENDPOINTS (Mock for now) =============

@app.post("/api/tts/synthesize")
async def synthesize_speech(text: str):
    """
    TTS: Text → Audio
    """
    try:
        # TODO: Implement real TTS when ready
        audio_path = "/static/sample-audio.wav"
        
        return {
            "status": "success",
            "audio_path": audio_path
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============= TRAINING ENDPOINTS =============

training_state = {
    "status": "idle",
    "model": None,
    "progress": 0
}

@app.get("/api/training/status")
async def get_training_status():
    """Get training status"""
    return training_state

@app.post("/api/training/start/{model_type}")
async def start_training(model_type: str):
    """Start training"""
    try:
        global training_state
        training_state["status"] = "running"
        training_state["model"] = model_type
        
        if model_type == "llm":
            return {
                "status": "success",
                "message": f"{model_type} training started"
            }
        
        return {
            "status": "warning",
            "message": f"Training for {model_type} not yet implemented"
        }
    except Exception as e:
        training_state["status"] = "failed"
        return {
            "status": "error",
            "message": str(e)
        }

# ============= CONFIG ENDPOINTS =============

@app.get("/api/config/{model_type}")
async def get_config(model_type: str):
    """Get model configuration"""
    try:
        if model_type == "llm":
            return llm_config
        return {"error": "Unknown model type"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "🚀"*30)
    print("Starting Ooredoo AI Framework")
    print("🚀"*30 + "\n")
    print("📍 API: http://localhost:8000")
    print("🌐 Web: http://localhost:8000/")
    print("\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")