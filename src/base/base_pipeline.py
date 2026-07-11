# src/base/base_pipeline.py
from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """Classe abstraite pour tous les pipelines"""
    
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        print(f"✅ {self.name} initialized")
    
    @abstractmethod
    def load(self):
        """Load data (can be from HuggingFace or local)"""
        pass
    
    @abstractmethod
    def split(self, dataset):
        """Split dataset"""
        pass
    
    @abstractmethod
    def silver(self, dataset):
        """Silver layer"""
        pass
    
    @abstractmethod
    def gold(self, dataset):
        """Gold layer"""
        pass
    
    def process_full(self):
        """Run full pipeline"""
        print(f"🔄 {self.name} pipeline starting...")
        raw = self.load()
        splits = self.split(raw)
        
        train_silver = self.silver(splits.get("train"))
        val_silver = self.silver(splits.get("val"))
        train_gold = self.gold(train_silver)
        val_gold = self.gold(val_silver)
        
        print(f"✅ {self.name} pipeline completed!")
        return {
            "train_gold": train_gold,
            "val_gold": val_gold,
            "splits": splits
        }