# src/pipelines/stt_pipeline.py

from src.base.base_pipeline import BasePipeline
from datasets import load_dataset
from transformers import AutoProcessor
import librosa

class STPipeline(BasePipeline):
    """Pipeline للـ STT"""
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(
            config.get('model_id', 'facebook/wav2vec2-large-xlsr-53-arabic')
        )
    
    def load(self, data_path):
        """Load audio data"""
        dataset = load_dataset("audiofold", data_dir=data_path)
        print(f"✅ STT Bronze: {len(dataset)} files")
        return dataset
    
    def split(self, dataset):
        """Split train/test"""
        return dataset.train_test_split(test_size=0.1)
    
    def silver(self, dataset):
        """Validate audio"""
        def validate(ex):
            try:
                audio, sr = librosa.load(ex['file'], sr=16000)
                duration = len(audio) / 16000
                if duration < 1 or duration > 15:
                    return False
                return True
            except:
                return False
        
        return dataset.filter(validate)
    
    def gold(self, dataset):
        """Tokenize audio"""
        def preprocess(ex):
            audio, sr = librosa.load(ex['file'], sr=16000)
            processed = self.processor(
                audio=audio,
                text=ex.get('text', ''),
                sampling_rate=16000
            )
            return {
                "input_values": processed["input_values"][0],
                "attention_mask": processed["attention_mask"][0],
                "labels": processed.get("input_ids", [])
            }
        
        return dataset.map(preprocess)