import argparse
import json
import optuna
import sys
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, load_metric
from elasticsearch import Elasticsearch
from optuna.integration import WeightsAndBiasesCallback
from optuna.samplers import MultivariateTPESampler
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)

wandb.init(project="LLaMAM-Pipeline-Enhanced")

class LLaMATrainingPipeline:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.metrics = {"rouge": load_metric("rouge"), "meteor": load_metric("meteor")}
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    def create_and_ingest_index(self, index_name, filepath):
        self.es.indices.create(index=index_name, ignore=400)
        with open(filepath, 'r') as f:
            data = json.load(f)
            for i, doc in enumerate(data):
                self.es.index(index=index_name, id=i, body=doc)

    def load_dataset(self, dataset_name, download):
        if download:
            self.dataset_train = load_dataset(dataset_name, split="train")
            self.dataset_valid = load_dataset(dataset_name, split="validation")
        self.dataset = concatenate_datasets([self.dataset_train, self.dataset_valid])

    def tokenize_function(self, examples):
        return self.tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, dim=-1)
        predictions_text = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge = self.metrics['rouge'].compute(predictions=predictions_text, references=labels_text)
        meteor = self.metrics['meteor'].compute(predictions=predictions_text, references=labels_text)
        return {"rouge": rouge, "meteor": meteor}

def objective(trial):
    wandb.init()
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    model_path = trial.suggest_categorical("model_path", ["bert-base-uncased", "roberta-base"])
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0, 0.1)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
    adam_epsilon = trial.suggest_float("adam_epsilon", 1e-9, 1e-7)
    adam_beta1 = trial.suggest_float("adam_beta1", 0.8, 0.95)
    adam_beta2 = trial.suggest_float("adam_beta2", 0.98, 0.999)

    pipeline = LLaMATrainingPipeline(model_path, "cuda")
    pipeline.load_dataset("squad", download=True)
    tokenized_dataset = pipeline.dataset.map(pipeline.tokenize_function, batched=True, num_proc=4)

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="rouge",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=learning_rate,
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="wandb",
        push_to_hub=False,
        deepspeed="ds_config.json",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.05)]
    )

    optimizer = AdamW(params=pipeline.model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(tokenized_dataset) // per_device_train_batch_size
    )

    trainer = Trainer(
        model=pipeline.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    'labels': torch.stack([f['labels'] for f in data])},
        compute_metrics=pipeline.compute_metrics,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator="ddp"
    )

    trainer.train()
    torch.cuda.empty_cache()

    eval_dataset = load_dataset("squad", split="test")
    eval_result = trainer.evaluate(eval_dataset)
    wandb.log(eval_result)
    return eval_result["eval_rouge"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--download_dataset', type=bool, default=False)
    args = parser.parse_args()

    pipeline = LLaMATrainingPipeline("distilbert-base-uncased", "cuda")
    pipeline.create_and_ingest_index(args.index_name, args.filepath)

    study = optuna.create_study(direction="maximize", sampler=MultivariateTPESampler(), callbacks=[WeightsAndBiasesCallback()])
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
