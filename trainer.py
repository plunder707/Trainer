import argparse
import logging
import sys
import torch
from datasets import load_dataset, concatenate_datasets, load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.pipeline import ExtractiveQAPipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)

class LLaMATrainingPipeline:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.set_device(device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.metrics = {
            "rouge": load_metric("rouge"),
            "meteor": load_metric("meteor")
        }

        # Initialize Elasticsearch Document Store
        self.document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
        
        # Initialize Haystack Retriever
        self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model=model_path, use_gpu=True)

    def set_device(self, device):
        self.device = device
        if self.device not in ["cuda", "cpu"]:
            logging.error("Invalid device specified. Using CPU.")
            self.device = "cpu"

    def load_dataset(self, dataset_name="squad"):
        # Assuming you've already loaded your data into Elasticsearch
        # Retrieve your data here if needed
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

def data_collator(data):
    return {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data])
    }

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_pipeline = LLaMATrainingPipeline(args.model_path, device)

    training_pipeline.load_dataset(args.dataset_name)
    tokenized_dataset = training_pipeline.dataset.map(training_pipeline.tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="rouge",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True if device == "cuda" else False
    )

    trainer = Trainer(
        model=training_pipeline.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        compute_metrics=training_pipeline.compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--dataset_name', type=str, default="squad", help='Name of the dataset to use')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save training results')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warm-up steps for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--logging_dir', type=str, default="./logs", help='Directory for logs')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
    parser.add_argument('--save_total_limit', type=int, default=5, help='Total limit of saved models')
    args = parser.parse_args()

    main(args)
