# Fine-tuning RoBERTa-Large on MRPC with LoRA and MeZO+LoRA

[![CI Status](https://img.shields.io/badge/CI-Passing-green)](https://github.com/alex-kuznetsov7677/MeZO_7677)  
*Repository fork of [MeZO](https://github.com/princeton-nlp/MeZO) with LoRA+MeZO experiments.*

---
## Project Overview
This project implements memory-efficient fine-tuning of RoBERTa-Large using:
- **LoRA** (Low-Rank Adaptation)
- **MeZO** (Memory-efficient Zeroth-order Optimization) 
on the MRPC (Microsoft Research Paraphrase Corpus) dataset.

---

## Installatiom  
```bash
git clone https://github.com/alex-kuznetsov7677/MeZO_7677
cd MeZO
pip install -r requirements.txt
git checkout feature/lora-mezo2

---

## Dataset Preparation  
```bash  
python medium_models/tools/generate_full_data_MRPC.py

---
## Training Commands

./medium_models/run_lora.sh \
  --task MRPC \
  --model roberta-large \
  --lora_r 8 \
  --max_steps 1000

./medium_models/run_mezo_lora.sh \
  --task MRPC \
  --model roberta-large \
  --zero_order_eps 0.1 \
  --max_steps 1000

---

## Key Modifications


### MRPC-Specific Tokenization
For paraphrase classification (MRPC), we implemented a specialized tokenization scheme:
<s> sentence1 </s></s> sentence2 </s>

### Class Weight Balancing:
class_weights = torch.tensor( )

### Architecture Enforcement:
"architectures": ["RobertaForSequenceClassification"] 

---

## Results Summary

### 1. LoRA (Adam) Fine-Tuning
| Metric          | Range       |
|-----------------|-------------|
| F1-Score    | 0.84–0.86     |
| Accuracy    | 0.81–0.84     |
| Time        | ~3 hours (CPU)|
| Memory      | ~4 GB      |

### 2. MeZO+LoRA Fine-Tuning
| Metric          | Range       |
|-----------------|-------------|
| F1-Score    | 0.8		|
| Accuracy    | 0.67            |
| Time        | ~4-6 hours (CPU)|
| Memory      | ~2 GB           |


---

## MeZO+LoRA Performance Analysis

**Key Findings:**
- LoRA: Achieved stable convergence with F1=89.1, validating the pipeline.
- The MeZO+LoRA combination failed to achieve meaningful learning, consistently predicting the majority class (67% baseline accuracy)
- All tested mitigation strategies proved ineffective for this task
- The combination of noisy gradient estimates (MeZO) and low-rank updates (LoRA) creates additive negative effects that destabilize training.
- MeZO typically requires 3-5x more training steps for convergence compared to standard methods (as per original paper).
- The recommended packet size of 1024 is difficult to achieve for CPU configurations, which significantly affects the quality of the gradient estimation.

---

**Attempted Improvements:**

1. **Hyperparameter Tuning**  
   Tested ranges:
   - Learning rates: 1e-6 to 1e-3
   - Perturbation scale: 1e-4 to 1e-1  
   - LoRA ranks: 4 to 8 
   - Batch sizes: 4 to 32  (the batch size of 64 leads to long calculations on the CPU)

2. **Class Imbalance Solutions**  
   Implemented without success:
   - Oversampling minority class (33%->50%)
   - Class weighting (weights=[1.5358, 0.7414] or [2, 0.5])

---

**Conclusions:**
1. Gradient estimation noise from MeZO too severe
2. LoRA's low-rank updates may amplify forward-pass noise

---

**Suggested Directions:**
- Larger batch sizes (>64) for stable gradients
- Layer-specific perturbation scaling
- Undersampling majority class (65% → 50%)
- Dynamic data reweighting during training
- Gradient accumulation with smaller batches
- L2 regularization
- Gradient clipping


---

## Repository Structure

MeZO_7677/
└ medium_models/
    ├ tools/
    │   └ generate_full_data_MRPC.py  # Скрипт для подготовки датасета MRPC
    ├ finetune_loramezo.sh            # Скрипт для обучения MeZO + LoRA
    └ finetune_lora.sh                # Скрипт для классического LoRA обучения


---


🔗 Pull Request: MeZO#2
📧 Contact: [a.kuznetsov7677@gmail.com]
