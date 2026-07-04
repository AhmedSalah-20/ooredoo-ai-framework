# src/base/base_pipeline.py
from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """Classe abstraite pour tous les pipelines (STT, LLM, TTS)"""
    
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        print(f"✅ {self.name} initialized")
    
    @abstractmethod
    def load(self, data_path):
        """Bronze: Load données brutes"""
        pass
    
    @abstractmethod
    def split(self, dataset):
        """Split train/test"""
        pass
    
    @abstractmethod
    def silver(self, dataset):
        """Silver: Format/Validate"""
        pass
    
    @abstractmethod
    def gold(self, dataset):
        """Gold: Preprocess final"""
        pass
    
    def process_full(self, data_path):
        """Full pipeline: Bronze→Silver→Gold"""
        print(f"🔄 {self.name} pipeline starting...")
        raw = self.load(data_path)
        split = self.split(raw)
        train_silver = self.silver(split['train'])
        test_silver = self.silver(split['test'])
        train_gold = self.gold(train_silver)
        test_gold = self.gold(test_silver)
        print(f"✅ {self.name} pipeline complete!")
        return train_gold, test_gold