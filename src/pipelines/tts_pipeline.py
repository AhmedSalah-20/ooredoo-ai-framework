import yaml
from datasets import load_dataset, Audio
import numpy as np

class TTSPipeline:
    def __init__(self, config=None, config_path="configs/tts_config.yaml"):
        if config is not None:
            self.config = config
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                
        self.dataset_config = self.config.get("dataset", {})
        self.col_text = "sentence"
        self.col_audio = "audio"

    def detect_format(self, columns):
        text_candidates = ["sentence", "text", "transcription", "normalized_text"]
        audio_candidates = ["audio", "speech", "wav"]
        
        for c in text_candidates:
            if c in columns:
                self.col_text = c
                break
        for c in audio_candidates:
            if c in columns:
                self.col_audio = c
                break
        print(f"🔍 Auto-detected TTS Format - Text: '{self.col_text}', Audio: '{self.col_audio}'")

    def load_data(self):
        dataset_path = self.dataset_config.get("path", "mozilla-foundation/common_voice_13_0")
        print(f"📥 Loading TTS dataset: {dataset_path}")
        
        train_split = self.dataset_config.get("train_split", "train")
        ds_train = load_dataset(dataset_path, split=train_split)
        
        try:
            ds_test = load_dataset(dataset_path, split="test")
        except:
            ds_test = None
            
        self.detect_format(ds_train.column_names)
        
        target_sr = self.dataset_config.get("target_sample_rate", 22050)
        ds_train = ds_train.cast_column(self.col_audio, Audio(sampling_rate=target_sr))
        if ds_test:
            ds_test = ds_test.cast_column(self.col_audio, Audio(sampling_rate=target_sr))
            
        return {"train": ds_train, "test": ds_test}

    def extract_audio_info(self, audio_val):
        """
        Safely extracts (numpy_array, sampling_rate) from various audio formats,
        handling standard dicts and torchcodec / PyTorch AudioDecoder objects.
        """
        if audio_val is None:
            return None, None
            
        # 1. Hugging Face Audio dict format (Standard)
        if isinstance(audio_val, dict):
            return audio_val.get("array"), audio_val.get("sampling_rate")
            
        # 2. torchcodec / PyTorch AudioDecoder format
        try:
            sr = None
            # Extract sampling rate
            if hasattr(audio_val, "metadata"):
                meta = audio_val.metadata
                sr = getattr(meta, "sample_rate", getattr(meta, "sampling_rate", None))
            if sr is None and hasattr(audio_val, "get_metadata"):
                meta = audio_val.get_metadata()
                sr = getattr(meta, "sample_rate", getattr(meta, "sampling_rate", None))
            if sr is None:
                sr = getattr(audio_val, "sample_rate", getattr(audio_val, "sampling_rate", 22050))
                
            # Extract the raw waveform array
            arr = None
            for method_name in ["get_all_audio_frames", "read_all", "get_audio_frames"]:
                if hasattr(audio_val, method_name):
                    try:
                        method = getattr(audio_val, method_name)
                        frames = method()
                        # Convert PyTorch tensor to numpy if needed
                        if hasattr(frames, "numpy"):
                            arr = frames.numpy()
                        elif hasattr(frames, "detach"):
                            arr = frames.detach().cpu().numpy()
                        else:
                            arr = frames
                        if arr is not None:
                            break
                    except Exception:
                        continue
            
            # Direct fallback
            if arr is None:
                if hasattr(audio_val, "numpy"):
                    arr = audio_val.numpy()
                elif hasattr(audio_val, "detach"):
                    arr = audio_val.detach().cpu().numpy()

            if arr is not None and sr is not None:
                arr = np.asarray(arr, dtype=np.float32)
                # Convert to mono if it's stereo
                if arr.ndim > 1:
                    if arr.shape[0] < arr.shape[1]:
                        arr = arr[0]
                    else:
                        arr = arr[:, 0]
                return arr, int(sr)
                
        except Exception as e:
            print(f"⚠️ Failed to dynamically extract audio: {e}")
            
        return None, None

    def process_silver(self, dataset):
        print("🧼 Silver layer: Filtering empty text and missing audio...")
        def filter_valid(example):
            # 1. Vérifier si le texte existe et n'est pas vide
            text_val = example.get(self.col_text)
            has_text = bool(text_val and str(text_val).strip())
            
            # 2. Vérifier juste que la colonne audio n'est pas vide (sans forcer le décodage)
            audio_val = example.get(self.col_audio)
            has_audio = audio_val is not None
            
            return has_text and has_audio
            
        return dataset.filter(filter_valid)

    def process_gold(self, dataset):
        print("🏆 Gold layer: Normalizing text for TTS...")
        def normalize(example):
            text = str(example[self.col_text]).strip()
            return {"tts_normalized_text": text}
        return dataset.map(normalize)
        
    def run(self):
        raw_data = self.load_data()
        val_size = self.dataset_config.get("val_size", 300)
        
        splits = raw_data["train"].train_test_split(test_size=val_size, seed=42)
        
        train_silver = self.process_silver(splits["train"])
        val_silver = self.process_silver(splits["test"])
        
        train_gold = self.process_gold(train_silver)
        val_gold = self.process_gold(val_silver)
        
        return {
            "raw": raw_data,
            "silver": {"train": train_silver, "val": val_silver},
            "gold": {"train": train_gold, "val": val_gold},
            "cols": {"text": self.col_text, "audio": self.col_audio}
        }