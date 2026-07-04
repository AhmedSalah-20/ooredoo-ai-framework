# src/trainers/llm_trainer.py
# ← ADAPTE DE TON FILE EXISTANT!

from src.base.base_trainer import BaseTrainer
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType

class LLMTrainer(BaseTrainer):
    """Fine-tuning engine pour LLM - hérite de BaseTrainer"""
    
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
    
    def _setup_environment(self):
        """🔥 Configure l'environnement GPU"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        print(f"✅ GPU: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}")
    
    def _load_quantization_config(self):
        """📥 Crée la config 4-bit quantization"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self):
        """📥 Charge le modèle en 4-bit (économe en RAM)"""
        self._setup_environment()
        
        print(f"📥 Chargement: {self.config['model']['model_id']}")
        
        bnb_config = self._load_quantization_config()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_id'],
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_id']
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Modèle chargé")
    
    def _get_lora_config(self):
        """⚙️ Configure les adapteurs LoRA"""
        lora_cfg = self.config['lora']
        return LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['alpha'],
            lora_dropout=lora_cfg['dropout'],
            target_modules=lora_cfg['target_modules'],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
    def _get_training_arguments(self):
        """📊 Configure les arguments de training"""
        train_cfg = self.config['training']
        
        return TrainingArguments(
            output_dir=train_cfg['output_dir'],
            learning_rate=float(train_cfg['learning_rate']),
            per_device_train_batch_size=int(train_cfg['batch_size']),
            gradient_accumulation_steps=int(train_cfg['gradient_accumulation_steps']),
            num_train_epochs=int(train_cfg['epochs']),
            save_steps=int(train_cfg['save_steps']),
            logging_steps=int(train_cfg['logging_steps']),
            optim="paged_adamw_32bit",
            fp16=False,
            bf16=False,
            eval_strategy="steps",
            eval_steps=int(train_cfg['save_steps']),
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            max_grad_norm=1.0,
            gradient_checkpointing=True,
        )
    
    def train(self, train_dataset, eval_dataset):
        """🚀 Lance le fine-tuning"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("❌ engine.load_model() d'abord!")
        
        print("=" * 70)
        print("🔥 LANCEMENT DU FINE-TUNING")
        print("=" * 70)
        
        training_args = self._get_training_arguments()
        lora_config = self._get_lora_config()
        
        print("🔥 SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=training_args,
        )
        
        print("🚀 Training...")
        trainer.train()
        
        print("=" * 70)
        print("✅ FINE-TUNING TERMINÉ!")
        print("=" * 70)
        
        return trainer
    
    def save_model(self, output_path):
        """💾 Sauvegarde le modèle et tokenizer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("❌ Modèle pas chargé")
        
        print(f"💾 Saving: {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✅ Done!")
    
    def inference(self, text):
        """💬 Inférence - génère du texte"""
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        result = pipe(text, max_length=100)
        return result[0]["generated_text"]