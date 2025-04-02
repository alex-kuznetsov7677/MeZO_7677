import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent 
medium_models_path = root_dir / "medium_models"
sys.path.append(str(medium_models_path))

# Теперь можно импортировать
from src.trainer import Trainer
import torch
import os 
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Union, Any, Optional
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

from peft import PeftModel
from datasets import load_dataset, DatasetDict
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch.utils.data import DataLoader 
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
import torch
from torch.utils.data import WeightedRandomSampler



class PairDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        sep_present = (batch['input_ids'] == self.tokenizer.sep_token_id).any(dim=1)
        if not sep_present.all():
            print("No SEP!")
        return batch




class MeZOTrainingArguments(TrainingArguments):
    def __init__(
        self,
        *args,
        zero_order_optim: bool = False,
        zero_order_eps: float = 1e-3,
        lora_optim: bool = False,  
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.zero_order_optim = zero_order_optim
        self.zero_order_eps = zero_order_eps
        self.lora_optim = lora_optim

class LoRAMeZOTrainer(Trainer):

    def __init__(self, *args, lora_config=None, **kwargs):
        label_names = kwargs.pop('label_names', None)
        super().__init__(*args, **kwargs)
 
        self.metrics_files = {
            'loss': open('loss_log.txt', 'w'),
            'accuracy': open('accuracy_log.txt', 'w'),
            'f1': open('f1_log.txt', 'w')
        }

        if label_names:
            self.label_names = label_names

        if lora_config and not self._is_lora_initialized():
            self._initialize_lora(lora_config)
        
        if self.args.lora_optim:
            self._freeze_non_lora_params()

    def log_metrics(self, step, metrics):
        if 'loss' in metrics:
            self.metrics_files['loss'].write(f"{metrics['loss']},")
        if 'accuracy' in metrics:
            self.metrics_files['accuracy'].write(f"{metrics['accuracy']},")
        if 'f1' in metrics:
            self.metrics_files['f1'].write(f"{metrics['f1']},")
        

        for file in self.metrics_files.values():
            file.flush()
            

    def _is_lora_initialized(self):
        return any('lora_' in n for n, _ in self.model.named_parameters())

    def _initialize_lora(self, lora_config):

        from peft import get_peft_model 
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _freeze_non_lora_params(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = 'lora_' in name

    def should_optim(self, name, param):

        if self.args.lora_optim:
            return 'lora_' in name and param.requires_grad
        return super().should_optim(name, param)
    def zo_forward(self, model, inputs):
        #model.eval()
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        return loss
    def training_step(self, model, inputs, num_items_in_batch=None):  # Добавляем num_items_in_batch
        inputs = {k: v.to(model.device) for k, v in inputs.items()}


        if not self.args.zero_order_optim or not self.args.lora_optim:
            return super().training_step(model, inputs)
    
        lora_params = [(n, p) for n, p in model.named_parameters() 
                      if self.should_optim(n, p)]

        original = {n: p.detach().clone() for n, p in lora_params}
        z = {}
    

        for n, p in lora_params:
            z[n] = torch.randn_like(p)

            p.data.add_(z[n]*self.args.zero_order_eps)  
    
        loss_plus = self.zo_forward(model, inputs)
    

        for n, p in lora_params:
            p.data.sub_(z[n]*2*self.args.zero_order_eps)  
    
        loss_minus = self.zo_forward(model, inputs)
    

        with torch.no_grad():
            grad_estimate = (loss_plus - loss_minus) / (2 * self.args.zero_order_eps)
            grads = [grad_estimate * z[n] for n, p in lora_params]

            updates = []
            for (n, p), grad in zip(lora_params, grads):
                update = -self.args.learning_rate * (grad + self.args.weight_decay * p.data)
                updates.append(update)


            total_update_norm = torch.norm(torch.stack([torch.norm(u, 2) for u in updates]), 2)
            max_allowed_norm = 0.1 
    
            if total_update_norm > max_allowed_norm:
                clip_coef = max_allowed_norm / (total_update_norm + 1e-6)
                updates = [u * clip_coef for u in updates]


            for (n, p), update in zip(lora_params, updates):
                p.data.copy_(original[n])
                p.data.add_(update)

                    


        with torch.no_grad():
            for (n, p), grad in zip(lora_params, grads):
                if hasattr(p, 'grad'):
                    del p.grad
                torch.cuda.empty_cache()
            del grads, grad_estimate, z
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        if self.state.global_step % 1 == 0:
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == inputs['labels']).float().mean()
                f1 = f1_score(inputs['labels'].cpu(), preds.cpu(), average='binary')
            
                metrics = {
                    'loss': (loss_plus + loss_minus).item() / 2,
                    'accuracy': acc.item(),
                    'f1': f1
                }
                self.log_metrics(self.state.global_step, metrics)
        if self.state.global_step % 10 == 0:
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        return (loss_plus + loss_minus) / 2  
    def __del__(self):
        for file in self.metrics_files.values():
            file.close()


            
class BalancedLoRAMeZOTrainer(LoRAMeZOTrainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def get_train_dataloader(self):
        if self.class_weights is not None:
            class_counts = np.bincount(self.train_dataset["labels"])
            sample_weights = torch.tensor(
                [self.class_weights[label] for label in self.train_dataset["labels"]],
                dtype=torch.float32
            )
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers
            )
        return super().get_train_dataloader()
def main():
    torch.manual_seed(42)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_dataset("glue", "mrpc")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    

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
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_dataset.set_format("torch")


    
    class_counts = np.bincount(tokenized_dataset["train"]["labels"])
    total_samples = sum(class_counts)
    class_weights = total_samples / (2. * class_counts)


    sample_weights = class_weights[tokenized_dataset["train"]["labels"]]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(tokenized_dataset["train"]),
        replacement=True 
    )


    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"  
    )
    

    model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-large").to(device)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    training_args = MeZOTrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=16,
        learning_rate=1e-3,
        zero_order_optim=True,
        weight_decay=0.01, 
        lora_optim=True,
        zero_order_eps=1e-2,
        max_steps=1000,
        remove_unused_columns=False 
    )


    data_collator = PairDataCollator(tokenizer=tokenizer,
                                       padding= "max_length",
                                       max_length=128
                                     )

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler, 
        collate_fn=data_collator,
        num_workers=4
    )

    
    trainer = BalancedLoRAMeZOTrainer(
        model=model,
        args=training_args,
        class_weights=class_weights,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.label_names = ["labels"]

    
    trainer.train()


if __name__ == "__main__":
    main()
