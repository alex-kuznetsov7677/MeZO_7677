import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
import logging
from tqdm import tqdm
import time
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import RandomSampler
from itertools import cycle
from transformers import get_scheduler
import evaluate
from lion_pytorch import Lion
from collections import Counter


from help_logging import (
    setup_logging,
    clear_log_files,
    format_time,
    get_memory_stats
)

class PairDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)

        sep_present = (batch['input_ids'] == self.tokenizer.sep_token_id).any(dim=1)
        if not sep_present.all():
            print("No SEP!")
        return batch
def format_time(seconds):

    return str(timedelta(seconds=seconds))
def main():
    loss_logger, memory_logger, main_logger, console_logger = setup_logging()
    torch.manual_seed(42)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "batch_size": 16,
        "desired_steps": 1000,
        "lora_rank": 8,
        "learning_rate": 1e-4,
    }

    dataset = load_dataset("glue", "mrpc")["train"]

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    label_counts = Counter(dataset["label"])

    def tokenize_function(examples):
        return tokenizer(
            text=examples["sentence1"],
            text_pair=examples["sentence2"],
            truncation=True,
            max_length=128,
            padding=False,
            return_token_type_ids=False
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_dataset.set_format("torch")
    


    total = sum(label_counts.values())
    class_weights = torch.tensor([total/label_counts[0], total/label_counts[1]], dtype=torch.float32).to(device)

    dataset_size = len(tokenized_dataset)
    steps_per_epoch = dataset_size // config["batch_size"]
    metric = evaluate.load("glue", "mrpc")

    
    data_collator = PairDataCollator(tokenizer=tokenizer,
                                     padding= "max_length",
                                     max_length=128
                                     )
    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=config["batch_size"],
        collate_fn=data_collator,
        shuffle=True,
        num_workers=4
    )

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=2
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=config["lora_rank"],
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        task_type="SEQ_CLS"
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    
    model.train()
    progress_bar = tqdm(range(config["desired_steps"]), desc="Training")
    step=0
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=config["desired_steps"]
    )
    for step in range(config["desired_steps"]):

        if step % steps_per_epoch == 0:
            train_iter = iter(DataLoader(
                tokenized_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4,
                collate_fn=data_collator
            )) 
    
    
        batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
            
        loss_logger.info(f"{loss.item():.6f}")
        _, _, peak_memory = get_memory_stats()
        memory_logger.info(f"{peak_memory:.2f}")            



    eval_results = metric.compute()
    print("\nFinal Results:")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"F1-score: {eval_results['f1']:.4f}")
    print(f"Class weights used: {class_weights.cpu().numpy()}")

    
if __name__ == "__main__":
    main()
