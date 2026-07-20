import re
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
from src.base.base_pipeline import BasePipeline

class STTPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        model_config = config["model"]
        
        print(f"✅ Loading Whisper processor: {model_config['model_id']}")
        self.processor = WhisperProcessor.from_pretrained(
            model_config["model_id"],
            language=model_config.get("language", "ar"),
            task=model_config.get("task", "transcribe")
        )

    def load(self):
        dataset_config = self.config["dataset"]
        source = dataset_config.get("source", "huggingface")
        path = dataset_config.get("path")
        
        print(f"📥 Loading dataset from {source.upper()}: {path}")

        try:
            if source == "huggingface":
                # Protection: Ken user 7at URL kemla bl ghalet, nndhafouha
                if path.startswith("http"):
                    path = path.replace("https://huggingface.co/datasets/", "").replace("http://huggingface.co/datasets/", "")
                
                train = load_dataset(path, split=dataset_config.get("train_split", "train"), streaming=dataset_config.get("streaming", False))
                try:
                    test = load_dataset(path, split=dataset_config.get("test_split", "test"), streaming=dataset_config.get("streaming", False))
                except ValueError:
                    test = None  # Famma datasets ma fihomch test
            else:
                # ================= LOCAL FILES & OTHER SOURCES =================
                if source == "jsonl":
                    ds = load_dataset("json", data_files=path)
                elif source == "csv":
                    ds = load_dataset("csv", data_files=path)
                elif source == "parquet":
                    ds = load_dataset("parquet", data_files=path)
                elif source == "audio_folder":
                    # HuggingFace yaqra dossier kemel mtaa audio w ykharej metadata
                    ds = load_dataset("audiofolder", data_dir=path)
                else:
                    raise ValueError(f"Source {source} n'est pas supportée.")

                train = ds.get("train")
                test = ds.get("test")

            # Protection: Ken l'dataset fiha ken train (kima barcha fichiers CSV/JSON), na9smouha houni
            if test is None:
                print("⚠️ No test split found in source. Creating an automatic 10% test split...")
                splits = train.train_test_split(test_size=0.1, seed=42)
                train = splits["train"]
                test = splits["test"]

            return {"train": train, "test": test}

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise e

    def split(self, dataset):
        val_size = self.config["dataset"].get("val_size", 200)
        print(f"✂️ Splitting dataset - val_size={val_size}")

        if hasattr(dataset["train"], "train_test_split"):
            result = dataset["train"].train_test_split(test_size=val_size, seed=42)
            return {
                "train": result["train"],
                "val": result["test"],
                "test": dataset["test"]
            }
        else:
            # Streaming mode
            val = dataset["train"].take(val_size)
            train = dataset["train"].skip(val_size)
            return {"train": train, "val": val, "test": dataset["test"]}

    def silver(self, dataset):
        print("🧼 Silver layer - cleaning...")

        cfg = self.config["dataset"]
        text_col = cfg["text_column"]
        remove_punct = cfg.get("remove_punctuation", True)
        lower_case = cfg.get("lowercase", True)
        # 1. Fonction ll Filtrage en BATCH
        def _filter_batch(batch):
            # Trajaa True ken text fih ktiba, False ken vide
            return [
                bool(text and len(str(text).strip()) > 0) 
                for text in batch[text_col]
            ]
        
        # 2. Fonction ll Nettoyage en BATCH
        def _clean_text_batch(batch):
            cleaned = []
            for text in batch[text_col]:
                val = str(text) if text is not None else ""
                if remove_punct:
                    val = re.sub(r'[,\?\.\!\-\;\:\"\%“”]', '', val)
                if lower_case:
                    val = val.lower()
                cleaned.append(val.strip())
            
            # Modifiw ken text column, w nrajjou batch
            batch[text_col] = cleaned
            return batch
        # 3. L'Execution w l'astuce mtaa batched=True
        print("⏳ Filtering empty transcripts...")
        dataset = dataset.filter(_filter_batch, batched=True)
        
        print("⏳ Cleaning text...")
        dataset = dataset.map(_clean_text_batch, batched=True)
        
        return dataset


    def gold(self, dataset):
        print("🏆 Gold layer - Preparing features and labels...")
        
        cfg = self.config["dataset"]
        audio_col = cfg.get("audio_column", "audio")
        text_col = cfg.get("text_column", "transcript")
        
        # 1. Audio Resampling to 16kHz (ضروري لـ Whisper)
        # الكود هذا يضمن إنو أي صوت يجي، يترد 16000 Hz مهما كان الـ source
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate=16000))
        
        # 2. Preparation function
        def _prepare_dataset(batch):
            audio = batch[audio_col]
            
            # Extract input features from audio
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            
            # Tokenize targets (text)
            batch["labels"] = self.processor.tokenizer(
                batch[text_col],
                truncation=True,
                max_length=self.config["model"].get("max_label_length", 448)
            ).input_ids
            
            return batch

        # 3. Apply transformation
        # نخدمو بالـ map باش نحضرو الـ features و الـ labels لكل الـ dataset
        dataset = dataset.map(_prepare_dataset, remove_columns=dataset.column_names)
        
        # 4. Final Duration Filter (للحماية من الـ OOM Errors)
        # Whisper ما يحبش فوق الـ 30 ثانية
        def _filter_duration(batch):
            # الـ input_features طولها يمثل مدة الصوت
            # نحسبو المدة: len(features) * hop_length / sampling_rate
            audio_length = len(batch["input_features"][0]) * 320 / 16000 
            return audio_length < 30.0
            
        dataset = dataset.filter(_filter_duration)
        
        print(f"✅ Gold layer ready! Dataset size: {len(dataset)}")
        return dataset 

