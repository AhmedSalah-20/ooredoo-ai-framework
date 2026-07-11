import yaml
from datasets import load_dataset

class LLMPipeline:
    def __init__(self, config_path="configs/llm_config.yaml"):
        print(f"⚙️ Initializing LLM Pipeline from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.dataset_config = self.config.get("dataset", {})
        
        # Variables dynamiques bech nkhabiw fihom l'format
        self.format_type = "unknown"
        self.col_prompt = "instruction"
        self.col_response = "output"

    def detect_format(self, columns):
        """Fonction intelligente bech ta3ref l'format mtaa l'dataset wa7adha"""
        if "conversations" in columns:
            self.format_type = "sharegpt"
            self.col_prompt = "conversations"
        elif "messages" in columns:
            self.format_type = "openai"
            self.col_prompt = "messages"
        elif "instruction" in columns and "output" in columns:
            self.format_type = "alpaca"
            self.col_prompt, self.col_response = "instruction", "output"
        elif "prompt" in columns and "response" in columns:
            self.format_type = "standard"
            self.col_prompt, self.col_response = "prompt", "response"
        elif "question" in columns and "answer" in columns:
            self.format_type = "qa"
            self.col_prompt, self.col_response = "question", "answer"
        
        print(f"🔍 Auto-detected Dataset Format: {self.format_type.upper()}")

    def load_data(self, dataset_path=None):
        hf_path = dataset_path or self.dataset_config.get("huggingface")
        print(f"📥 Loading LLM dataset: {hf_path}")
        
        train_split = self.dataset_config.get("train_split", "train")
        test_split = self.dataset_config.get("test_split", "test")
        
        ds_train = load_dataset(hf_path, split=train_split)
        try:
            ds_test = load_dataset(hf_path, split=test_split)
        except Exception:
            ds_test = None
            
        # Ndetectiw l'format mel colonnes mtaa Train
        self.detect_format(ds_train.column_names)
            
        return {"train": ds_train, "test": ds_test}

    def extract_prompt_response(self, example):
        """Tkharej l'Prompt w Response mahma ken l'format (Yesta3melha l'Gold Layer w l'API)"""
        prompt, response = "", ""
        
        if self.format_type in ["sharegpt", "openai"]:
            conv = example.get(self.col_prompt, [])
            if isinstance(conv, list) and len(conv) >= 2:
                # .get("value") lel ShareGPT, w .get("content") lel OpenAI
                prompt = conv[0].get("value", conv[0].get("content", ""))
                response = conv[1].get("value", conv[1].get("content", ""))
        else:
            # Pour Alpaca, Standard, w QA
            prompt = example.get(self.col_prompt, "")
            # Ken fama 'input' f'Alpaca nzidouh
            if self.format_type == "alpaca" and example.get("input"):
                prompt += "\n" + str(example.get("input"))
            response = example.get(self.col_response, "")
            
        return prompt, response

    def process_silver(self, dataset):
        print("🧼 Silver layer: Cleaning Text & Filtering empty prompts...")
        
        def filter_empty(example):
            prompt, response = self.extract_prompt_response(example)
            return bool(prompt and str(prompt).strip() and response and str(response).strip())
            
        return dataset.filter(filter_empty)

    def process_gold(self, dataset):
        print("🏆 Gold layer: Formatting to ChatML...")
        
        def format_chatml(example):
            prompt, response = self.extract_prompt_response(example)
            formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            return {"text_formatted": formatted_text}
            
        cols_to_remove = [c for c in dataset.column_names if c != "text_formatted"]
        return dataset.map(format_chatml, remove_columns=cols_to_remove)
        
    def run(self, dataset_path=None):
        raw_data = self.load_data(dataset_path)
        val_size = self.dataset_config.get("val_size", 300)
        
        splits = raw_data["train"].train_test_split(test_size=val_size, seed=42)
        train_data = splits["train"]
        val_data = splits["test"]
        
        train_silver = self.process_silver(train_data)
        val_silver = self.process_silver(val_data)
        
        train_gold = self.process_gold(train_silver)
        val_gold = self.process_gold(val_silver)
        
        return {
            "raw": {"train": raw_data["train"], "test": raw_data["test"]},
            "silver": {"train": train_silver, "val": val_silver},
            "gold": {"train": train_gold, "val": val_gold}
        }