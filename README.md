# 🔐 Network Intrusion Detection with Adversarial Robustness
### CIC-IDS 2017 · Random Forest · AdvGAN · Adversarial Retraining

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20PhD-purple)

---

## Overview

This repository contains the full pipeline for a **network intrusion detection system (IDS)** built on the [CIC-IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset, extended with **adversarial attack generation** using AdvGAN and **adversarial retraining** to restore model robustness.

The work is part of ongoing PhD research on adversarial robustness in federated intrusion detection systems.

### Key Results

| Stage | Metric | Value |
|---|---|---|
| Baseline Random Forest | F1 Score (attacks) | **0.99** |
| AdvGAN (black-box attack) | Evasion Success Rate | **45.06%** |
| After Adversarial Retraining | Evasion Success Rate | **0.00%** |

> A model with F1 = 0.99 on clean data is **not robust** — nearly half of adversarial flows evade detection. Adversarial retraining completely eliminates evasion with no accuracy loss.

---

## Pipeline

```
CIC-IDS 2017 Dataset (2.83M flows · 79 features · 15 labels)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  Remove ±Inf/negatives · Correlation filter
│   79 → 43 features  │  RobustScaler · Variance thresholding
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Random Forest IDS  │  Binary: BENIGN vs. ATTACK
│  F1 = 0.99          │  80/20 stratified split
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  AdvGAN Training    │  WGAN loss + Black-box evasion loss
│  ESR = 45.06%       │  Generator · Discriminator · 30 epochs
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Adversarial         │  Augment training set with adversarial samples
│ Retraining          │  ESR drops to 0.00%
└─────────────────────┘
```

---

## Repository Structure

```
cic-ids-advgan/
│
├── CIC_IDS2017_Analysis.ipynb   # Main notebook (all sections)
│
├── models/                      # Saved model artifacts (auto-created)
│   ├── advgan_generator.pth
│   ├── advgan_discriminator.pth
│   ├── rf_baseline.joblib
│   ├── rf_robust_baseline.joblib
│   ├── robust_scaler.joblib
│   └── feature_columns.joblib
│
├── figures/                     # Pipeline diagram & plots
│   └── pipeline_flowchart.png
│
├── presentation/
│   └── presentation.tex         # Beamer slides (Overleaf-ready)
│
├── requirements.txt
└── README.md
```

---

## Notebook Structure

| Section | Description |
|---|---|
| **1. Imports & Configuration** | All dependencies, device setup, dataset path |
| **2. Data Loading** | Explore files, concatenate CSVs, class distribution |
| **3. Preprocessing** | Cleaning, correlation filter, scaling, variance thresholding |
| **4. Baseline Classifier** | Random Forest training and evaluation |
| **5. AdvGAN** | Architecture, training loop, evasion evaluation, perturbation analysis |
| **6. Adversarial Retraining** | Data augmentation, robust classifier, re-evaluation |
| **7. Model Persistence** | Save and reload all trained components |

---

## Installation

```bash
git clone https://github.com/yourname/cic-ids-advgan.git
cd cic-ids-advgan
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
torch>=2.0
joblib>=1.3
```

---

## Dataset Setup

1. Download the **CIC-IDS 2017** dataset from the [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract the `MachineLearningCVE` folder
3. Update the `DATA_FOLDER` path in **Section 1** of the notebook:

```python
DATA_FOLDER = "/path/to/MachineLearningCVE"
```

The folder should contain these 8 CSV files:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```

---

## AdvGAN Design

### Generator
Takes a real attack sample `x` and produces an adversarial version:

```
x_adv = x + G(x) · mask · ε
```

- `G(x)` is bounded to `[-1, 1]` via Tanh
- `mask` controls which features can be perturbed (all 1s by default)
- `ε = 0.05` controls attack strength

### Loss Function

```
L_D = -E[D(real)] + E[D(fake)]              # WGAN critic loss
L_G = -E[D(G(x))] + α · E[1 - P_RF(BENIGN)]  # Generator loss
```

- `α = 10` balances realism vs. evasion
- `P_RF(BENIGN)` is queried from the Random Forest in **black-box** mode — no gradient access to the target model

### Top Perturbed Features

| Rank | Feature | Mean \|Δ\| |
|---|---|---|
| 1 | Idle Std | 0.0157 |
| 2 | Max Packet Length | 0.0113 |
| 3 | Active Std | 0.0102 |
| 4 | Bwd Packet Length Max | 0.0102 |
| 5 | Bwd Packets/s | 0.0098 |

Perturbations are small (< 2% of feature range) yet highly effective — and the adversarial samples preserve the correlation structure of real attacks.

---

## Reproducing Results

Run the notebook end-to-end:

```bash
jupyter notebook CIC_IDS2017_Analysis.ipynb
```

Or run sections individually — each section is self-contained with clear inputs and outputs.

To skip retraining and load saved models directly, jump to **Section 7.2 (Load Models)**.

---

## Research Context

This work is part of a broader PhD project on **adversarial robustness in federated intrusion detection systems**, combining:

- DRL-based client selection for federated learning
- Adversarial attack generation targeting distributed IDS models
- Robustness evaluation under varying federation scenarios

### Next Steps

- [ ] Multi-class attack classification (15 labels)
- [ ] Learned feature mutability mask (domain-constrained perturbations)
- [ ] Transfer attacks across model families (XGBoost, LSTM)
- [ ] Integration into federated learning pipeline
- [ ] Evaluation on CICIDS 2018 and UNSW-NB15

---

## Citation

If you use this code or pipeline in your research, please cite:

```bibtex
@misc{yourname2025cicids,
  author    = {Your Name},
  title     = {Network Intrusion Detection with Adversarial Robustness: CIC-IDS 2017},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/yourname/cic-ids-advgan}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>PhD Research · Department of Computer Science · University Name</sub>
</div>
