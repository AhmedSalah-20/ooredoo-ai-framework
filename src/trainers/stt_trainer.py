# src/trainers/stt_trainer.py

from src.base.base_trainer import BaseTrainer
import torch
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    TrainingArguments,
    Trainer
)

class STTTrainer(BaseTrainer):
    """Fine-tune Wav2Vec2 للـ STT"""
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
    
    def load_model(self):
        """Load Wav2Vec2 model"""
        print(f"📥 Loading STT: {self.config.get('model_id', 'wav2vec2')}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.get('model_id', 'facebook/wav2vec2-large-xlsr-53-arabic')
        )
        
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.config.get('model_id', 'facebook/wav2vec2-large-xlsr-53-arabic'),
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
        ).to(self.device)
        
        print(f"✅ STT Model loaded")
    
    def train(self, train_dataset, eval_dataset):
        """Train STT"""
        print("🚀 STT Training...")
        
        training_args = TrainingArguments(
            output_dir="./models/stt-model",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            fp16=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        print("✅ STT training complete!")
        return trainer
    
    def save_model(self, output_path):
        """Save STT model"""
        print(f"💾 Saving STT: {output_path}")
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        print(f"✅ Saved!")
    
    def inference(self, audio_path):
        """STT inference"""
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription