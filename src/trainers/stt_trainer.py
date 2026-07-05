import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.base.base_trainer import BaseTrainer


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class STTTrainer(BaseTrainer):
    """Fine-tuning Whisper avec LoRA ou QLoRA (choix 100% via config, pas de code à toucher)."""

    def __init__(self, config):
        super().__init__(config)
        self.processor = WhisperProcessor.from_pretrained(
            config['model']['model_id'],
            language=config['model'].get('language', 'Arabic'),
            task=config['model'].get('task', 'transcribe'),
        )
        self.metric = evaluate.load("wer")

    def load_model(self):
        peft_cfg = self.config.get('peft', {})
        quant = peft_cfg.get('quantization', 'none')  # none | 8bit | 4bit (QLoRA réelle)

        quant_config = None
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"🤖 Chargement Whisper ({self.config['model']['model_id']}), quantization={quant}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config['model']['model_id'],
            quantization_config=quant_config,
            device_map="auto" if quant_config else None,
        )
        if quant_config is None:
            self.model.to(self.device)

        # Whisper: évite les hallucinations de tokens de langue/tâche forcés
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []

        if peft_cfg.get('use_lora', True):
            if quant_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=peft_cfg.get('r', 8),
                lora_alpha=peft_cfg.get('alpha', 32),
                target_modules=peft_cfg.get('target_modules', ["q_proj", "v_proj"]),
                lora_dropout=peft_cfg.get('dropout', 0.05),
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        print("✅ Modèle STT prêt")

    def compute_metrics(self, pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train(self, train_dataset, eval_dataset):
        cfg = self.config['training']
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg['output_dir'],
            per_device_train_batch_size=cfg['batch_size'],
            per_device_eval_batch_size=cfg['batch_size'],
            gradient_accumulation_steps=cfg.get('grad_accum_steps', 1),
            learning_rate=float(cfg['learning_rate']),  # float() nécessaire: YAML lit "1e-5" comme string
            warmup_steps=cfg.get('warmup_steps', 500),
            max_steps=cfg['max_steps'],
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",  # anciennes versions de transformers: "evaluation_strategy"
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=cfg.get('save_steps', 500),
            eval_steps=cfg.get('eval_steps', 500),
            logging_steps=cfg.get('logging_steps', 25),
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            report_to=["none"],
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            compute_metrics=self.compute_metrics,
            
           
        )

        print("🚀 Lancement entraînement STT...")
        trainer.train()
        print("✅ Entraînement terminé")
        return trainer

    def save_model(self, output_path):
        print(f"💾 Sauvegarde: {output_path}")
        self.model.save_pretrained(output_path)  # ne sauvegarde que l'adapter LoRA (léger) si PEFT actif
        self.processor.save_pretrained(output_path)

    def inference(self, audio_path):
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.to(self.model.device))
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
