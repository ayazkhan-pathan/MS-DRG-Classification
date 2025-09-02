# 🧠 DRG-GPT

**Predicting Diagnosis-Related Group (DRG) Codes from Discharge Summaries Using Large Language Models**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![LLM](https://img.shields.io/badge/Model-LLaMA%203%20%7C%20BioClinicalBERT%20%7C%20Custom%20Heads-green.svg)

---

## 📌 Overview

**DRG-GPT** is a research project focused on automating the classification of **Diagnosis-Related Group (DRG)** codes from **hospital discharge summaries** using **Large Language Models (LLMs)**.

We started from the [DRG-LLaMA](https://github.com/hanyin88/DRG-LLaMA) repository as a **baseline** and extended it with several **new fine-tuning strategies** and **prediction techniques**.

Our goal is to improve **accuracy**, **interpretability**, and **scalability** for **clinical coding automation**.

---

## 🚀 Key Features

* **Baseline DRG-LLaMA Evaluation**

  * Started from the original DRG-LLaMA setup as a foundation.
* **Chunk-Based Training** ✅ *(New)*

  * Long discharge summaries are split into **overlapping sentence-based chunks**.
  * Trained Bert-based models on chunk-level representations.
* **Chunk-Based Prediction** ✅ *(New)*

  * Aggregates chunk-level logits for **document-level DRG predictions**.
* **Summarization-Based Classification** ✅ *(New)*

  * Experimented with summarizing discharge summaries before classification.
* **Custom Classification Heads** ✅ *(New)*

  * Designed lightweight classification heads on top of **LLaMA 3**.
* **Multi-Model Experimentation**

  * Compared **LLaMA 3**, **BioClinicalBERT**, and **custom architectures**.
* **Ongoing Research**

  * More methods are currently being tested and will be uploaded in future commits.

---

## 🏗️ Project Structure

```bash
DRG-GPT/
│── data/                 # Datasets (MIMIC-IV Discharge Summaries, preprocessed)
│── models/               # Fine-tuned checkpoints & custom heads
│── notebooks/            # Exploratory analysis, testing, experiments
│── src/
│   ├── preprocessing/    # Custom cleaning, abbreviation expansion, chunking
│   ├── training/         # Fine-tuning LLaMA & other models
│   ├── evaluation/       # Metrics, confusion matrices, interpretability
│── results/              # PR curves, ROC curves, metrics, summaries
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
```

---

## 🧪 Methodology

### **1. Baseline: DRG-LLaMA**

* Started from [DRG-LLaMA](https://github.com/hanyin88/DRG-LLaMA) baseline.
* Evaluated base performance on raw discharge summaries.

### **2. Our Contributions**

| **Technique**               | **Description**                                        | **Status**     |
| --------------------------- | ------------------------------------------------------ | -------------- |
| Chunk-Based Training        | Overlapping sentence chunks for LLaMA fine-tuning      | ✅ Done         |
| Chunk-Based Prediction      | Logit normalization for document-level DRG codes       | ✅ Done         |
| Summarization Approach      | Summarized long discharge summaries pre-classification | ✅ Done         |
| Custom Classification Heads | Lightweight MLP-style heads on LLaMA                   | ✅ Done         |
| Hyperparameter Sweeps       | Optimized LR, batch size, dropout, and LoRA configs    | ✅ Ongoing      |
| LLaMA 3 Fine-Tuning         | Testing on larger datasets                             | 🔄 In Progress |

---

## 📊 Evaluation Metrics

* **Primary Metrics**

  * Micro F1-score
  * Macro F1-score
  * ROC-AUC per DRG code
* **Secondary Metrics**

  * Precision / Recall @ Top-K
  * Model confidence calibration
* **Interpretability**

  * Token-level contributions
  * Key phrase extraction for clinical auditing

---

## 🛠️ Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/iScape4code/DRG-GPT.git
cd DRG-GPT
```

### **2. Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 📚 Usage

### **1. Preprocess the Dataset**

```bash
python src/preprocessing/clean_and_chunk.py \
    --input data/discharge_summaries.csv \
    --output data/processed_chunks.csv
```

### **2. Fine-Tune the Model**

```bash
python src/training/train_llama.py \
    --model llama3 \
    --epochs 5 \
    --batch_size 8 \
    --lr 2e-5
```

### **3. Evaluate the Model**

```bash
python src/evaluation/eval_model.py \
    --model models/llama3_best.pt \
    --test data/test_set.csv
```

---

## 🔗 Dependency on DRG-LLaMA

This project builds on the research and baseline methods from [DRG-LLaMA](https://github.com/hanyin88/DRG-LLaMA).
Since DRG-LLaMA **does not specify a license**, we **do not redistribute their code** here.

If you want to reproduce the baseline results, please clone DRG-LLaMA separately:

```bash
git clone https://github.com/hanyin88/DRG-LLaMA.git
```

---

## 📅 Roadmap

* [x] Evaluate DRG-LLaMA baseline
* [x] Implement chunk-based training and prediction
* [x] Experiment with summarization-based classification
* [x] Add custom classification heads
* [ ] Fine-tune LLaMA 3 on larger datasets
* [ ] Evaluate BioClinicalBERT vs LLaMA 3
* [ ] Release comparison benchmarks
* [ ] Deploy as an interactive API

---

## 📌 References

* [DRG-LLaMA Repository](https://github.com/hanyin88/DRG-LLaMA)
* MIMIC-IV Discharge Summaries Dataset (PhysioNet)
* [LLaMA 3](https://ai.meta.com/llama/)
* [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

---

## 🧑‍💻 Contributors

* **Mohammed Ayazkhan Pathan**
  *Data Science Research Assistant @ Western Sydney University*

---


