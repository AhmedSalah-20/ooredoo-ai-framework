# src/pipelines/tts_pipeline.py

from src.base.base_pipeline import BasePipeline
from datasets import Dataset
import json

class TTSPipeline(BasePipeline):
    """Pipeline للـ TTS"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def load(self, data_path):
        """Load text data"""
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        
        dataset = Dataset.from_dict({
            "text": [d["text"] for d in data],
            "speaker": [d.get("speaker", "default") for d in data]
        })
        print(f"✅ TTS Bronze: {len(dataset)} samples")
        return dataset
    
    def split(self, dataset):
        """Split train/test"""
        return dataset.train_test_split(test_size=0.1)
    
    def silver(self, dataset):
        """Validate text"""
        def validate(ex):
            if not ex['text'] or len(ex['text'].strip()) < 3:
                return False
            return True
        
        return dataset.filter(validate)
    
    def gold(self, dataset):
        """Prepare for TTS"""
        def prepare(ex):
            return {
                "text": ex["text"],
                "speaker_id": hash(ex["speaker"]) % 100
            }
        
        return dataset.map(prepare)