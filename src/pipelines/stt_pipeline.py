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
        print("🧼 Silver layer - cleaning text & filtering duration...")

        cfg = self.config["dataset"]
        text_col = cfg["text_column"]
        audio_col = cfg.get("audio_column", "audio")
        
        remove_punct = cfg.get("remove_punctuation", True)
        lower_case = cfg.get("lowercase", True)
        
        # 1. نجبدو الـ Min والـ Max من الـ Config اللي بعثها الـ UI
        min_dur = float(cfg.get("min_duration", 1.0))
        max_dur = float(cfg.get("max_duration", 20.0))

        # 2. تنظيف النص (نخليوها Batched باش تخدم فيسع)
        def _clean_text_batch(batch):
            cleaned = []
            for text in batch[text_col]:
                val = str(text) if text is not None else ""
                if remove_punct:
                    val = re.sub(r'[,\?\.\!\-\;\:\"\%“”،؟؛]', '', val)
                if lower_case:
                    val = val.lower()
                cleaned.append(val.strip())
            batch[text_col] = cleaned
            return batch

        print("⏳ Cleaning text...")
        dataset = dataset.map(_clean_text_batch, batched=True)

        # 3. فيلتر يثبت في النص والوقت مع بعضهم
        def _filter_valid(ex):
            # الشرط الأول: النص ميكونش فارغ
            text_valid = bool(ex[text_col] and len(str(ex[text_col]).strip()) > 0)
            
            # الشرط الثاني: طول الصوت بالثواني
            audio = ex[audio_col]
            duration = len(audio["array"]) / audio["sampling_rate"]
            dur_valid = (min_dur <= duration <= max_dur)
            
            return text_valid and dur_valid

        print(f"⏳ Filtering: Duration [{min_dur}s - {max_dur}s] and Empty Text...")
        dataset = dataset.filter(_filter_valid)

        return dataset


    def gold(self, dataset):
        print("🏆 Gold layer - Preparing features and labels...")
        
        cfg = self.config["dataset"]
        audio_col = cfg.get("audio_column", "audio")
        text_col = cfg.get("text_column", "transcript")
        
        # 1. Resampling 16kHz
        dataset = dataset.cast_column(audio_col, Audio(sampling_rate=16000))
        
        def _prepare_dataset(batch):
            audio = batch[audio_col]
            
            # Extract features
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            
            # Tokenize labels
            batch["labels"] = self.processor.tokenizer(
                batch[text_col],
                truncation=True,
                max_length=self.config["model"].get("max_label_length", 448)
            ).input_ids
            
            return batch

        # 2. Apply Mapping w نغطسو الـ Tensors (ونحينا الفيلتر الغالط)
        dataset = dataset.map(_prepare_dataset, remove_columns=dataset.column_names)
        
        print(f"✅ Gold layer ready! Dataset size: {len(dataset)}")
        return dataset
