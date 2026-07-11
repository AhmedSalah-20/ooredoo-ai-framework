# src/unified_pipeline.py - FIXED

import yaml
from src.pipelines.llm_pipeline import LLMPipeline
from src.trainers.llm_trainer import LLMTrainer
from datasets import load_from_disk
class UnifiedPipeline:
    """
    ORCHESTRATOR: Connecte STT → LLM → TTS
    """
    
    def __init__(self, base_config_path="configs/base_config.yaml"):
        # Load configs
        try:
            with open("configs/models/llm_config.yaml") as f:
                self.llm_config = yaml.safe_load(f)
            print("✅ Config loaded")
        except Exception as e:
            print(f"⚠️ Config error: {e}")
            self.llm_config = {}
    
    # ===== LLM PIPELINE =====
    def prepare_llm(self):
        """Prépare données LLM"""
        print("\n" + "="*70)
        print("📝 LLM DATA PIPELINE")
        print("="*70)
        
        llm_pipeline = LLMPipeline(self.llm_config)
        
        # Load
        raw = llm_pipeline.load(self.llm_config['data']['dataset_path'])
        
        # Silver
        train_silver = llm_pipeline.silver(raw['train'])
        test_silver = llm_pipeline.silver(raw['test'])
        
        # Gold
        train_gold = llm_pipeline.gold(train_silver)
        test_gold = llm_pipeline.gold(test_silver)
        
        # Save
        train_gold.save_to_disk("src/data/processed/llm_train")
        test_gold.save_to_disk("src/data/processed/llm_test")
        
        print(f"✅ LLM data prepared!")
        return train_gold, test_gold
    
    def train_llm(self, train_dataset=None, eval_dataset=None):
        """Fine-tune LLM"""
        print("\n" + "="*70)
        print("📝 LLM TRAINING")
        print("="*70)
        
        # Load datasets si pas fournis
        if train_dataset is None:
            train_dataset = load_from_disk("src/data/processed/llm_train")
        if eval_dataset is None:
            eval_dataset = load_from_disk("src/data/processed/llm_test")
        
        # Train
        trainer = LLMTrainer(self.llm_config)
        trainer.load_model()
        trainer.train(train_dataset, eval_dataset)
        trainer.save_model(self.llm_config['training']['output_dir'])
        
        print(f"✅ LLM trained!")
        return trainer
    
    def inference_llm(self, text):
        """LLM Inference"""
        trainer = LLMTrainer(self.llm_config)
        trainer.load_model()
        return trainer.inference(text)