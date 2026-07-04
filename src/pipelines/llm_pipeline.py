# src/pipelines/llm_pipeline.py
# ← ADAPTE DE TON FILE EXISTANT!

from src.base.base_pipeline import BasePipeline
from datasets import load_dataset
from transformers import AutoTokenizer

class LLMDataPipeline(BasePipeline):
    """Pipeline pour LLM - hérite de BasePipeline"""
    
    def __init__(self, config):
        super().__init__(config)
        
        model_name = config.get('model_id') if isinstance(config, dict) else config
        print(f"⚙️ Chargement du tokenizer : {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # ===== BRONZE: Load + Split =====
    def load(self, file_path):
        """Bronze: Load + Split"""
        raw_ds = load_dataset('json', data_files=file_path, split='train')
        test_size = self.config.get('test_size', 0.1) if isinstance(self.config, dict) else 0.1
        return raw_ds.train_test_split(test_size=test_size)
    
    def split(self, dataset):
        """Split déjà fait dans load() - retourne le dataset"""
        return dataset
    
    # ===== SILVER: Schema Safe Formatting =====
    def silver(self, dataset):
        """Silver: Format chat template"""
        def format(ex):
            instruction = ex.get("instruction") or ex.get("prompt")
            output = ex.get("output") or ex.get("response")
            if not instruction or not output: 
                return {"text": ""}
            return {"text": f"<|user|>{instruction}</s><|assistant|>{output}</s>"}
        
        return dataset.map(format, remove_columns=dataset.column_names).filter(lambda x: x["text"] != "")
    
    # ===== GOLD: Tokenization =====
    def gold(self, dataset):
        """Gold: Tokenize"""
        max_len = self.config.get('max_length', 512) if isinstance(self.config, dict) else 512
        
        def tok(ex):
            return self.tokenizer(
                ex["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_len
            )
        return dataset.map(tok, batched=True, remove_columns=["text"])