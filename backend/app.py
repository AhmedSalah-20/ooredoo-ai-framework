import sys
import io
import base64
import logging
from pathlib import Path
import yaml
import json
import os
import soundfile as sf
import smtplib
from email.message import EmailMessage

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

# Setup System Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.llm_pipeline import LLMPipeline
from src.pipelines.stt_pipeline import STTPipeline
from src.pipelines.tts_pipeline import TTSPipeline

# ==========================================
# 1. LOGGING & APP INITIALIZATION
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ooredoo AI Framework",
    description="STT + LLM + TTS Enterprise API",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates Static Setup
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


# ==========================================
# 2. DATABASE (JSON FILE) & GLOBAL STATES
# ==========================================
DB_FILE = "users_db.json"

def load_db():
    """تقرا اليوزرات من ملف الـ JSON، كان مفماش تصنع واحد فيه الـ Admin"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # par défaut
    default_db = {
        "directeur@ooredoo.tn": {"password": "admin", "role": "Administrateur", "status": "active"}
    }
    save_db(default_db)
    return default_db

def save_db(db_data):
    """تسجل التغييرات في ملف الـ JSON"""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=4)

# نعيطولها باش نعمرو الـ variable
USERS_DB = load_db()


# ==========================================
# 3. CONFIGURATION LOADING
# ==========================================
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


# ==========================================
# 4. PYDANTIC MODELS
# ==========================================
class PipelineRequest(BaseModel):
    source: str
    dataset_path: str
    output_dir: str = None


# ==========================================
# 5. DEPENDENCIES & HELPER FUNCTIONS
# ==========================================
def verify_admin(x_user_email: str = Header(None)):
    """تثبت هل الـ Email اللي بعث الـ Request عندو رول Admin في الـ DB"""
    if not x_user_email:
        raise HTTPException(status_code=401, detail="Action non autorisée. Email manquant.")
    
    user = USERS_DB.get(x_user_email)
    if not user or user["role"] != "Administrateur":
        raise HTTPException(status_code=403, detail="Accès refusé. Réservé à l'Administrateur.")
    
    return x_user_email


def audio_to_base64(audio_array, sampling_rate) -> str:
    """تحويل الـ Audio Array لـ Base64 لتبسيطه في الـ Frontend"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sampling_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def send_approval_email(user_email: str):
    """إرسال إيميل تفعيل الحساب"""
    print("🚀 ------------------------------------------------")
    print(f"📧 EMAIL SENT TO: {user_email}")
    print("Subject: Compte Ooredoo AI Framework Activé")
    print("Body: Bonjour, votre compte a été approuvé par l'Administrateur.")
    print("🚀 ------------------------------------------------")


# ==========================================
# 6. UI PAGES ROUTES (FRONTEND NAVIGATION)
# ==========================================
@app.get("/")
async def root():
    return RedirectResponse(url="/login")

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/dashboard")
async def dashboard(request: Request):
    response = templates.TemplateResponse(request=request, name="index.html")
    # نمنعو الـ Navigateur باش يعمل Cache للـ Dashboard لدواعي أمنية
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ==========================================
# 7. AUTHENTICATION & USER MANAGEMENT API
# ==========================================
@app.post("/api/signup")
async def signup(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    if email in USERS_DB:
        return JSONResponse(status_code=400, content={"message": "Email existe déjà!"})
    
    USERS_DB[email] = {
        "password": password,
        "role": role,
        "status": "pending"
    }
    
    # 💥 السطر هذا ضروري باش يسجل في الفيشيي
    save_db(USERS_DB) 
    
    logger.info(f"🆕 Nouvelle demande de compte : {email} ({role})")
    return JSONResponse(content={"message": "Demande envoyée avec succès à l'Administrateur."})


@app.post("/api/login")
async def login_api(email: str = Form(...), password: str = Form(...)):
    user = USERS_DB.get(email)
    
    if not user or user["password"] != password:
        return JSONResponse(status_code=401, content={"message": "Email ou mot de passe incorrect."})
    
    if user["status"] == "pending":
        return JSONResponse(status_code=403, content={"message": "Votre compte est en attente d'approbation."})
        
    return JSONResponse(content={"email": email, "role": user["role"], "message": "Login successful"})


# ==========================================
# 8. ADMIN APPROVAL PANEL API
# ==========================================
@app.get("/api/admin/pending")
async def get_pending_users(admin_email: str = Depends(verify_admin)):
    pending = [
        {"email": email, "role": data["role"]} 
        for email, data in USERS_DB.items() 
        if data["status"] == "pending"
    ]
    return {"pending_users": pending}


@app.post("/api/admin/approve")
async def approve_user(
    email: str = Form(...), 
    action: str = Form(...), 
    admin_email: str = Depends(verify_admin)
):
    if email not in USERS_DB:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    
    if action == "accept":
        USERS_DB[email]["status"] = "active"
        save_db(USERS_DB) # 💥 تسجيل
        send_approval_email(email)
        return {"message": f"✅ Compte {email} activé avec succès!"}
        
    elif action == "reject":
        del USERS_DB[email]
        save_db(USERS_DB) # 💥 تسجيل
        return {"message": f"❌ Demande {email} refusée et supprimée."}


# ==========================================
# 9. PIPELINES EXECUTION ROUTES (LLM/STT/TTS)
# ==========================================
@app.post("/api/pipeline/llm/process")
def process_llm_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 LLM Pipeline started for: {request.dataset_path}")
        pipeline = LLMPipeline(config_path="configs/llm_config.yaml")
        results = pipeline.run(dataset_path=request.dataset_path)
        
        samples = []
        silver_train = results["silver"]["train"]
        sample_subset = silver_train.select(range(min(3, len(silver_train))))
        
        for item in sample_subset:
            p, r = pipeline.extract_prompt_response(item)
            samples.append({"prompt": p, "response": r})
            
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
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/pipeline/stt/process")
def process_stt_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 STT Pipeline started for: {request.dataset_path}")
        
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
                    audio_b64 = audio_to_base64(audio_data["array"], audio_data["sampling_rate"])
                    samples_data.append({
                        "audio": f"data:audio/wav;base64,{audio_b64}",
                        "text": item[text_col]
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
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/pipeline/tts/process")
def process_tts_pipeline(request: PipelineRequest):
    try:
        logger.info(f"🚀 TTS Pipeline started for: {request.dataset_path}")
        
        current_config = tts_config.copy()
        if "dataset" not in current_config:
            current_config["dataset"] = {}
        current_config["dataset"]["source"] = request.source
        current_config["dataset"]["path"] = request.dataset_path

        pipeline = TTSPipeline(config=current_config)
        results = pipeline.run()
        
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

                arr, sr = pipeline.extract_audio_info(audio_data)
                if arr is None or sr is None:
                    continue

                audio_b64 = audio_to_base64(arr, sr)
                samples_data.append({
                    "audio": f"data:audio/wav;base64,{audio_b64}",
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
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    

# ==========================================
# 10. PLAYGROUND & TRAINING ENDPOINTS (MOCK)
# ==========================================
@app.get("/api/config/{model_type}")
async def get_config(model_type: str):
    if model_type == "llm":
        return llm_config
    elif model_type == "stt":
        return stt_config
    return JSONResponse(status_code=400, content={"error": "Unknown model type"})


@app.post("/api/stt/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return {"status": "success", "transcription": "أهلا وسهلا في منصة Ooredoo"}


@app.post("/api/tts/synthesize")
async def synthesize_speech(text: str):
    return {"status": "success", "audio_path": "/static/sample-audio.wav"}


@app.get("/api/training/status")
async def get_training_status():
    return training_state


@app.post("/api/training/start/{model_type}")
async def start_training(model_type: str):
    global training_state
    training_state["status"] = "running"
    training_state["model"] = model_type
    
    if model_type == "llm":
        return {"status": "success", "message": f"{model_type} training started"}
    return {"status": "warning", "message": f"Training for {model_type} not yet implemented"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


# ==========================================
# 11. MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "🚀"*30)
    print("Starting Ooredoo Enterprise AI Framework")
    print("🚀"*30 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")