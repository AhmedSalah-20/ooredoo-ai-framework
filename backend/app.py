import sys
import io
import base64
import logging
import yaml
from pathlib import Path
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.llm_pipeline import LLMPipeline
from src.pipelines.stt_pipeline import STTPipeline
from src.pipelines.tts_pipeline import TTSPipeline

# ============= LOGGING =============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= APP INIT =============
app = FastAPI(
    title="Ooredoo AI Framework",
    description="STT + LLM + TTS Enterprise API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# ============= CONFIG LOADING =============
# (Nesta3mlou dossier 'configs' kima tfehemna 9bila)
def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            logger.info(f"✅ Config loaded: {path}")
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"⚠️ Config warning for {path}: {e}")
        return {}

llm_config = load_config("configs/llm_config.yaml")
stt_config = load_config("configs/stt_config.yaml")
tts_config = load_config("configs/tts_config.yaml")

# ============= MODELS =============
class PipelineRequest(BaseModel):
    source: str
    dataset_path: str
    output_dir: str = None

# ============= GLOBAL TRAINING STATE =============
training_state = {
    "status": "idle",
    "model": None,
    "progress": 0
}


# ==========================================
#                  ROUTES
# ==========================================

@app.get("/")
async def root():
    """Serve main HTML page"""
    return FileResponse("frontend/static/index.html")

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/config/{model_type}")
async def get_config(model_type: str):
    if model_type == "llm":
        return llm_config
    elif model_type == "stt":
        return stt_config
    return {"error": "Unknown model type"}


# ==========================================
#              LLM PIPELINE
# ==========================================
@app.post("/api/pipeline/llm/process")
def process_llm_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 LLM Pipeline started for: {request.dataset_path}")
        
        # Initialisation mtaa l'pipeline (Dynamique)
        pipeline = LLMPipeline(config_path="configs/llm_config.yaml")
        results = pipeline.run(dataset_path=request.dataset_path)
        
        # Extraction des amthla b'tarika intelligente
        samples = []
        silver_train = results["silver"]["train"]
        sample_subset = silver_train.select(range(min(3, len(silver_train))))
        
        for item in sample_subset:
            p, r = pipeline.extract_prompt_response(item)
            samples.append({
                "prompt": p,
                "response": r
            })
            
        raw_test_len = len(results["raw"]["test"]) if results["raw"]["test"] is not None else 0
        
        return {
            "status": "success",
            "message": "LLM Pipeline executed successfully",
            "dataset": {
                "train_raw": len(results["raw"]["train"]),
                "test_raw": raw_test_len,
                "train_silver": len(results["silver"]["train"]),
                "val_silver": len(results["silver"]["val"]),
                "train_gold": len(results["gold"]["train"]),
                "val_gold": len(results["gold"]["val"])
            },
            "samples": samples
        }
    except Exception as e:
        logger.error(f"❌ LLM Pipeline error: {e}")
        return {"status": "error", "message": str(e)}


# ==========================================
#              STT PIPELINE
# ==========================================
@app.post("/api/pipeline/stt/process")
def process_stt_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 STT Pipeline started for: {request.dataset_path}")
        
        # Update config dynamically mel frontend request
        current_config = stt_config.copy()
        if "dataset" not in current_config:
            current_config["dataset"] = {}
            
        current_config["dataset"]["source"] = request.source
        current_config["dataset"]["path"] = request.dataset_path

        pipeline = STTPipeline(current_config)
        raw_dataset = pipeline.load()
        splits = pipeline.split(raw_dataset)
        
        train_silver = pipeline.silver(splits["train"])
        val_silver = pipeline.silver(splits["val"])
        
        train_gold = pipeline.gold(train_silver)
        val_gold = pipeline.gold(val_silver)

        # Extraction des Samples Audio
        samples_data = []
        try:
            logger.info("🎵 Extracting audio samples for frontend...")
            train_subset = raw_dataset["train"]
            sample_subset = train_subset.select(range(min(3, len(train_subset))))

            audio_col = current_config["dataset"].get("audio_column", "audio")
            text_col = current_config["dataset"].get("text_column", "sentence")

            for item in sample_subset:
                if audio_col in item and text_col in item:
                    audio_data = item[audio_col]
                    text_data = item[text_col]

                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        audio_data["array"],
                        audio_data["sampling_rate"],
                        format="WAV"
                    )
                    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    samples_data.append({
                        "audio": f"data:audio/wav;base64,{audio_base64}",
                        "text": text_data
                    })
        except Exception as e:
            logger.warning(f"⚠️ Could not extract audio samples: {e}")

        return {
            "status": "success",
            "message": "STT pipeline completed",
            "dataset": {
                "train_raw": len(raw_dataset["train"]),
                "test_raw": len(raw_dataset["test"]) if raw_dataset.get("test") else 0,
                "train_silver": len(train_silver),
                "val_silver": len(val_silver),
                "train_gold": len(train_gold),
                "val_gold": len(val_gold)
            },
            "samples": samples_data
        }
    except Exception as e:
        logger.error(f"❌ STT Pipeline error: {e}")
        return {"status": "error", "message": str(e)}

# ==========================================
#              TTS PIPELINE
# ==========================================
@app.post("/api/pipeline/tts/process")
def process_tts_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 TTS Pipeline started for: {request.dataset_path}")
        
        current_config = tts_config.copy()
        if "dataset" not in current_config:
            current_config["dataset"] = {}
            
        current_config["dataset"]["source"] = request.source
        current_config["dataset"]["path"] = request.dataset_path

        # Run Pipeline
        pipeline = TTSPipeline(config=current_config)
        results = pipeline.run()
        
        # Extraction des amthla (Samples) lel Frontend
        samples_data = []
        silver_train = results["silver"]["train"]
        sample_subset = silver_train.select(range(min(3, len(silver_train))))
        
        col_text = results["cols"]["text"]
        col_audio = results["cols"]["audio"]

        try:
            logger.info("🎵 Extracting TTS audio/text samples for frontend...")
            for item in sample_subset:
                text_data = item[col_text]
                audio_data = item[col_audio]

                # 💥 Use the safe dynamic extractor here
                arr, sr = pipeline.extract_audio_info(audio_data)

                if arr is None or sr is None:
                    logger.warning("⚠️ Skipping sample due to missing or undecodable audio array/rate")
                    continue

                buffer = io.BytesIO()
                sf.write(
                    buffer,
                    arr,
                    sr,
                    format="WAV"
                )
                audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                samples_data.append({
                    "audio": f"data:audio/wav;base64,{audio_base64}",
                    "text": text_data
                })
        except Exception as e:
            logger.warning(f"⚠️ Could not extract TTS samples: {e}")

        raw_test_len = len(results["raw"]["test"]) if results["raw"]["test"] is not None else 0
        
        return {
            "status": "success",
            "message": "TTS pipeline completed successfully",
            "dataset": {
                "train_raw": len(results["raw"]["train"]),
                "test_raw": raw_test_len,
                "train_silver": len(results["silver"]["train"]),
                "val_silver": len(results["silver"]["val"]),
                "train_gold": len(results["gold"]["train"]),
                "val_gold": len(results["gold"]["val"])
            },
            "samples": samples_data
        }
    except Exception as e:
        logger.error(f"❌ TTS Pipeline error: {e}")
        return {"status": "error", "message": str(e)}
    
# ==========================================
#          MOCK ENDPOINTS (STT / TTS)
# ==========================================
@app.post("/api/stt/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        mock_text = "أهلا وسهلا في منصة Ooredoo"
        return {"status": "success", "transcription": mock_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/tts/synthesize")
async def synthesize_speech(text: str):
    try:
        audio_path = "/static/sample-audio.wav"
        return {"status": "success", "audio_path": audio_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==========================================
#             TRAINING ENDPOINTS
# ==========================================
@app.get("/api/training/status")
async def get_training_status():
    return training_state

@app.post("/api/training/start/{model_type}")
async def start_training(model_type: str):
    global training_state
    try:
        training_state["status"] = "running"
        training_state["model"] = model_type
        
        if model_type == "llm":
            return {"status": "success", "message": f"{model_type} training started"}
            
        return {"status": "warning", "message": f"Training for {model_type} not yet implemented"}
    except Exception as e:
        training_state["status"] = "failed"
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "🚀"*30)
    print("Starting Ooredoo Enterprise AI Framework")
    print("🚀"*30 + "\n")
    print("📍 API: http://localhost:8000")
    print("🌐 Web: http://localhost:8000/")
    print("\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")