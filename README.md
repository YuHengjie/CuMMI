# CuMMI

**CuMMI (Curriculum-guided Multimodal Interaction Network)** is a curriculum-guided multimodal representation learning framework designed to infer **NPI (Nano–Protein Interaction)** across complex biological settings.

This repository provides the **supporting code for an ongoing research manuscript submission**.  
The code is released to promote transparency and reproducibility, and **research use is welcome**.

> 📌 **Citation**  
> If you find this work useful, please consider citing our paper.  
> **The formal citation information will be updated after publication.**

---

## Repository Structure

The repository is organized as follows:

CuMMI/
├── data/

├── text_embedding/

├── protein_embedding/

├── train_basic_10/

├── train_basic_30/

├── train_basic_50/

├── train_basic_75/

├── train_basic_100/

├── train_basic_only_protein/

├── train_basic_only_text/

├── train_date_10/

├── train_date_30/

├── train_date_50/

├── train_date_75/

├── train_date_100/

├── train_nano/

├── train_protein/

├── text_embedding_4_explain/

└── model_explain/

└── finetuning/


---

## Directory Description

### 📁 `data/`
This directory is used to store the dataset required for training, validation, and testing.  
This dataset could be downloaded by users following the paper description.

---

### 📁 `text_embedding/`
Implements text (tabular) feature encoding, which is used to transform structured experimental and contextual features into vector representations.

---

### 📁 `protein_embedding/`
Implements protein feature encoding, obtaining sequence and strucure representations for downstream multimodal learning.

---

### 📁 `train_basic_10`, `train_basic_30`, `train_basic_50`, `train_basic_75`, `train_basic_100`
These directories are used to construct and train the **CuMMI model** under a **curriculum learning strategy**.

- The numeric suffix indicates the **data proportion of Stage E** used during training.

---

### 📁 `train_basic_only_protein`
Training configuration using **only the protein modality**, for multimodal ablation analysis.

---

### 📁 `train_basic_only_text`
Training configuration using **only the text (tabular) modality**, for multimodal ablation analysis.

---

### 📁 `train_date_10`, `train_date_30`, `train_date_50`, `train_date_75`, `train_date_100`
These directories correspond to **date-based external validation and testing**.

- The numeric suffix indicates the **data proportion of Stage E** used during training.

---

### 📁 `train_nano`
External test setting based on **nanomaterial-held-out** splits, where nanomaterials appearing in the test set are not seen during training.

---

### 📁 `train_protein`
External test setting based on **protein-held-out** splits, where proteins in the test set are unseen during training.

---

### 📁 `text_embedding_4_explain`
Used for **feature ablation-based tabular feature importance analysis**.  
Text embeddings are generated specifically to support explainability experiments.

---

### 📁 `model_explain`
Contains model explanation modules for analyzing **model loss changes under feature ablation**, enabling interpretation of multimodal contributions.

---

### 📁 `finetuning`
Contains code for **fine-tuning** the model on controlled data splits, optimizing only the prediction head while freezing other layers to assess knowledge transfer from pretrained models.

---

## Notes

- This repository is intended **for research and academic use**. 
- **Commercial use of the Software is strictly prohibited**.
- The implementation reflects the experimental setup described in the manuscript.
- Some components may be further refined or reorganized after peer review.

---


**Thank you for your interest in CuMMI.**
