# 🎥 Video Captioning with S2VT + Attention  

This project implements a **Sequence-to-Sequence (S2VT) model with an attention mechanism** for automatic video caption generation using the **MSVD dataset**. The model takes pre-extracted video features as input and generates captions describing the video content.  

---

## 📂 Repository Structure  

- **`model_seq2seq.py`** – Main implementation containing:
  - Encoder and decoder models  
  - Data preprocessing functions  
  - End-to-end training/testing logic  

- **`hw2_seq2seq.sh`** – Bash script for testing and generating output captions (`testset_output.txt`).  

- **`TrainedModels/`** – Directory containing pre-trained models.  

- **`bleu_eval.py`** – Script for evaluating model accuracy using the BLEU metric.  

- **`training_data/` & `testing_data/`** – Directories where video features must be placed manually (not included in this repository).  

---

## 🏋️ Training

Open model_seq2seq.py.

**Set: Train = True inside the main_execution() function.**

Run the training script.

By default, the model is trained for 20 epochs.


## 🧪 Testing

**Open model_seq2seq.py and set: Train = False**

Run the bash testing script: **bash hw2_seq2seq.sh testing_data testset_output.txt**

The generated captions will be stored in **testset_output.txt.**

Evaluation results will be saved in **final_result.csv.**

## 📊 Results

Trained model: **model_batchsize_16_hiddensize_256_DP_0.3_worddim_2048.h5**

Achieved BLEU score: **0.6665**

BLEU scores are computed by comparing generated captions against ground-truth captions.

Results are logged in **final_result.csv.**

Generated captions are available in **testset_output.txt.**

## 🔖 Notes

The video feature folders (feat/) are not included in this repository. They must be added manually to **training_data/ and testing_data/.**

Ensure dataset preparation is completed before running training or testing.
