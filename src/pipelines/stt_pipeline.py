# src/pipelines/stt_pipeline.py
#
# Pipeline STT (Whisper) — Bronze -> Silver -> Gold, respecte le contrat BasePipeline
# (load / split / silver / gold) pour rester cohérent avec le reste du framework
# (même contrat que LLMDataPipeline).

import re
from src.base.base_pipeline import BasePipeline
from datasets import load_dataset, Audio
from transformers import WhisperProcessor


class STTPipeline(BasePipeline):
    """Pipeline STT pour le fine-tuning Whisper."""

    def __init__(self, config):
        super().__init__(config)
        self.processor = WhisperProcessor.from_pretrained(
            config['model']['model_id'],
            language=config['model'].get('language', 'Arabic'),
            task=config['model'].get('task', 'transcribe'),
        )

    def load(self, data_path=None):
        """Bronze: charge le dataset audio (HuggingFace en priorité, sinon local)."""
        ds_cfg = self.config['dataset']
        streaming = ds_cfg.get('streaming', False)

        if ds_cfg.get('huggingface'):
            train_ds = load_dataset(ds_cfg['huggingface'], split="train", streaming=streaming)
            test_ds = load_dataset(ds_cfg['huggingface'], split="test", streaming=streaming)
            # ⚠️ vérifie sur le dataset card que le split "test" existe bien sous ce nom
        else:
            local_path = ds_cfg['local_path']
            train_ds = load_dataset("audiofolder", data_dir=f"{local_path}/train")["train"]
            test_ds = load_dataset("audiofolder", data_dir=f"{local_path}/test")["train"]

        rate = ds_cfg.get('sampling_rate', 16000)
        train_ds = train_ds.cast_column('audio', Audio(sampling_rate=rate))
        test_ds = test_ds.cast_column('audio', Audio(sampling_rate=rate))
        return {"train": train_ds, "test": test_ds}

    def split(self, dataset):
        """Sépare train/val. En streaming: take/skip. Sinon: train_test_split classique."""
        ds_cfg = self.config['dataset']
        val_size = ds_cfg.get('val_size', 500)

        if ds_cfg.get('streaming', False):
            val = dataset['train'].take(val_size)
            train = dataset['train'].skip(val_size)
        else:
            parts = dataset['train'].train_test_split(test_size=val_size / max(len(dataset['train']), 1))
            train, val = parts['train'], parts['test']

        return {"train": train, "val": val, "test": dataset['test']}

    @staticmethod
    def _clean_text(ex):
        """Retire la ponctuation, uniformise la casse (utile pour les segments code-switchés FR/EN)."""
        ex["transcript"] = re.sub(r'[,\?\.\!\-\;\:\"\%“”]', '', ex["transcript"]).lower().strip()
        return ex

    def silver(self, dataset):
        """Filtre les audios trop courts/longs + transcripts vides, puis nettoie le texte."""
        def valid(ex):
            duration = len(ex['audio']['array']) / ex['audio']['sampling_rate']
            return 1.0 <= duration <= 20.0 and ex.get('transcript', '').strip() != ""

        return dataset.filter(valid).map(self._clean_text)

    def gold(self, dataset):
        """Extraction des features Whisper (log-mel) + tokenisation du texte."""
        def preprocess(ex):
            features = self.processor.feature_extractor(
                ex['audio']['array'], sampling_rate=ex['audio']['sampling_rate']
            ).input_features[0]
            labels = self.processor.tokenizer(ex['transcript']).input_ids
            return {"input_features": features, "labels": labels}

        return dataset.map(preprocess)