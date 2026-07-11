import torch


from transformers import (

    WhisperForConditionalGeneration,

    Seq2SeqTrainer,

    Seq2SeqTrainingArguments,

    DataCollatorForSeq2Seq

)



from peft import (

    LoraConfig,

    get_peft_model,

    prepare_model_for_kbit_training

)



import evaluate




class STTTrainer:



    def __init__(self, config):


        self.config = config



        model_cfg = config["model"]


        self.model = WhisperForConditionalGeneration.from_pretrained(

<<<<<<< HEAD
        # Whisper: évite les hallucinations de tokens de langue/tâche forcés
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []
=======
            model_cfg["model_id"],

            load_in_4bit=config["peft"].get(

                "use_4bit",

                False
>>>>>>> 815dabf (sttpipeline&llmpipeline)

            )

        )



        self.processor = None




    # ===================================
    # LoRA
    # ===================================


    def setup_peft(self):


        peft_cfg = self.config["peft"]



        if peft_cfg.get(

            "use_lora",

            False

        ):



            if peft_cfg.get(

                "use_4bit",

                False

            ):


                self.model = prepare_model_for_kbit_training(

                    self.model

                )




            lora_config = LoraConfig(


                r=peft_cfg["lora_r"],


                lora_alpha=peft_cfg["lora_alpha"],


                target_modules=peft_cfg["target_modules"],


                lora_dropout=peft_cfg["lora_dropout"],


                bias="none",


                task_type="SEQ_2_SEQ_LM"


            )



            self.model = get_peft_model(

                self.model,

                lora_config

            )



            self.model.print_trainable_parameters()




    # ===================================
    # Metrics
    # ===================================



    def compute_metrics(self,pred):


        wer_metric = evaluate.load(

            "wer"

        )


        predictions = pred.predictions


        labels = pred.label_ids



        decoded_preds = self.processor.batch_decode(

            predictions,

            skip_special_tokens=True

        )


        decoded_labels = self.processor.batch_decode(

            labels,

            skip_special_tokens=True

        )



        wer = wer_metric.compute(

            predictions=decoded_preds,

            references=decoded_labels

        )



        return {


            "wer": wer

        }




    # ===================================
    # TRAIN
    # ===================================


    def train(

        self,

        train_dataset,

        eval_dataset,

        processor

    ):



        self.processor = processor



        self.setup_peft()



        train_cfg = self.config["training"]




        args = Seq2SeqTrainingArguments(


            output_dir=train_cfg["output_dir"],


            num_train_epochs=train_cfg["num_train_epochs"],



            per_device_train_batch_size=train_cfg[

                "per_device_train_batch_size"

            ],



            per_device_eval_batch_size=train_cfg[

                "per_device_eval_batch_size"

            ],



            gradient_accumulation_steps=train_cfg[

                "gradient_accumulation_steps"

            ],



            learning_rate=train_cfg["learning_rate"],



            warmup_steps=train_cfg["warmup_steps"],



            fp16=train_cfg["fp16"],



            evaluation_strategy=train_cfg[

                "evaluation_strategy"

            ],



            eval_steps=train_cfg["eval_steps"],



            save_steps=train_cfg["save_steps"],



            logging_steps=train_cfg["logging_steps"],



            predict_with_generate=True,



            load_best_model_at_end=True,


            metric_for_best_model="wer",


            greater_is_better=False

        )





        data_collator = DataCollatorForSeq2Seq(

            tokenizer=processor.tokenizer,

            model=self.model

        )





        trainer = Seq2SeqTrainer(


            model=self.model,


            args=args,


            train_dataset=train_dataset,


            eval_dataset=eval_dataset,
<<<<<<< HEAD
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor),
            compute_metrics=self.compute_metrics,
            
           
=======


            data_collator=data_collator,


            compute_metrics=self.compute_metrics


>>>>>>> 815dabf (sttpipeline&llmpipeline)
        )



        trainer.train()


<<<<<<< HEAD
    def inference(self, audio_path):
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.to(self.model.device))
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
=======

        return trainer
>>>>>>> 815dabf (sttpipeline&llmpipeline)
