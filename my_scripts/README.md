# Fine-tuning RoBERTa-Large on MRPC with LoRA and MeZO+LoRA

[![CI Status](https://img.shields.io/badge/CI-Passing-green)](https://github.com/alex-kuznetsov7677/MeZO_7677)  
*Repository fork of [MeZO](https://github.com/princeton-nlp/MeZO) with LoRA+MeZO experiments.*

---

## Results Summary

### 1. LoRA (Adam) Fine-Tuning
| Metric          | Range       |
|-----------------|-------------|
| F1-Score    | 0.85–0.87   |
| Accuracy    | 0.82–0.85   |
| Time        | ~25 mins    |
| Memory      | ~4 GB      |

### 2. MeZO+LoRA Fine-Tuning
| Metric          | Range       |
|-----------------|-------------|
| F1-Score    | 0-0.2		|
| Accuracy    | 0.50–0.52   |
| Time        | ~60 mins    |
| Memory      | ~1.2 GB       |

---

## Hypotheses for MeZO+LoRA Failure
1. Noisy gradient estimates from zeroth-order optimization.
2. Hyperparameter sensitivity (e.g., step size, perturbation scale).
3. Compatibility issues between MeZO and LoRA's base model.
4. Insufficient training steps.

---

## Repository Structure

├── my_scripts/
│ ├── lora_trainer.py # LoRA + Adam script
│ ├── lora_mezo_trainer.py # MeZO + LoRA script
│ └── README.md # This report
└── requirements.txt # Dependencies


---

## Reproduction Guide
```bash
# Clone repo
git clone https://github.com/alex-kuznetsov7677/MeZO.git
cd MeZO
git checkout feature/lora-mezo

# Install dependencies
pip install -r requirements.txt

# Run LoRA+Adam
python my_scripts/lora_trainer.py

# Run MeZO+LoRA
python my_scripts/lora_mezo_trainer.py
2. Run Experiments
LoRA (Adam):


## Conclusion
✅ LoRA+Adam works as expected, achieving strong performance.

❌ MeZO+LoRA fails to converge under current settings.


🔗 Pull Request: MeZO#1
📧 Contact: [a.kuznetsov7677@gmail.com]