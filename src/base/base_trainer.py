# src/base/base_trainer.py
from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    """Classe abstraite pour tous les trainers (STT, LLM, TTS)"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ {self.__class__.__name__} initialized")
    
    @abstractmethod
    def load_model(self):
        """Load le modèle"""
        pass
    
    @abstractmethod
    def train(self, train_dataset, eval_dataset):
        """Lance l'entraînement"""
        pass
    
    @abstractmethod
    def save_model(self, output_path):
        """Sauvegarde le modèle"""
        pass
    
    def inference(self, input_data):
        """Méthode d'inférence (override dans subclasses)"""
        raise NotImplementedError()