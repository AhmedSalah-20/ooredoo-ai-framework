import yaml
from transformers import WhisperProcessor
from src.data.stt_loader import STTDataLoader

class STTPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r") as file:
            self.config_model = yaml.safe_load(file)['model']

        print("🧠 [Pipeline] Initialisation mté3 WhisperProcessor...")
        self.processor = WhisperProcessor.from_pretrained(
            self.config_model['model_id'],
            language=self.config_model['language'],
            task=self.config_model['task']
        )

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = self.processor.tokenizer(batch["transcript"]).input_ids
        return batch

    def run(self):
        """L'exécution complète mté3 l'pipeline"""
        # 1. Njibou l'Loader
        loader = STTDataLoader(self.config_path)
        splits = loader.load_and_split()

        processed_splits = {}
        print("🚀 [Pipeline] Lancement mté3 l'extraction des features...")
        
        for split_name, dataset in splits.items():
            print(f"   -> Traitement mté3 {split_name}...")
            # 2. Preprocessing pur (Audio & Text)
            ds = loader.preprocess(dataset)
            # 3. Traitement mté3 l'modèle (Features & Labels)
            ds = ds.map(self.prepare_dataset)
            
            processed_splits[split_name] = ds
            
        print("✅ [Pipeline] Data 7adhra 100% lel Fine-Tuning!")
        return processed_splits