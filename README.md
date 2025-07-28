# DeepLearning Combined Work

A curated collection of end‑to‑end PyTorch notebooks covering computer vision, NLP, time‑series forecasting, anomaly detection, and Transformers. These notebooks are self‑contained, reproducible, and GPU‑accelerated.

> ⚠️ **All rights reserved.** This repository is for portfolio and learning reference only. Do **not** reproduce, modify, or redistribute any part of this work without explicit permission.

---

## 📂 Notebooks

| Notebook                                | Task                                                                                   |
|-----------------------------------------|----------------------------------------------------------------------------------------|
| **1_CNNs_VGG_ResNet.ipynb**             | From‑scratch VGG‑16 (Version C) & ResNet‑18 on 64×64 images (dogs, cars, food); full data exploration, augmentation, training, metrics & confusion matrices. |
| **2_CNNs_ResNeXt.ipynb**                | Custom ResNeXt‑style CNN variants with kernel/pooling experiments and comparative benchmarks.     |
| **3_Vanishing_Gradient_Study.ipynb**    | “VGG‑Deep” gradient‑norm hooks to visualize vanishing gradients; compare with ResNet & small‑CNN variants. |
| **4_TimeSeries_RNN_Forecasting.ipynb**  | LSTM/GRU forecasting on AirQualityUCI (CO levels) with windowing, scaling, hyperparameter tuning & MAE/RMSE/R² metrics. |
| **5_Sentiment_LSTM_GRU.ipynb**          | IMDb sentiment analysis using LSTM & GRU; includes full preprocessing, embeddings, and precision/recall/F1 evaluation. |
| **6_Autoencoders_Anomaly_Detection.ipynb** | Dense, Conv1D & LSTM autoencoders for anomaly detection on EC2 CPU utilization; reconstruction‑error thresholding & ROC analysis. |
| **7_Transformer_From_Scratch.ipynb**    | Transformer encoder–decoder built from scratch (multi‑head self‑attention, positional encoding, masking) trained on a toy summarization task. |
| **8_Summarization_BillSum.ipynb**       | Fine‑tuned BART on the BillSum legislative summarization dataset; evaluated with ROUGE, BLEU & BERTScore. |
| **9_Summarization_MultiNews.ipynb**     | Fine‑tuned BART on the Multi‑News dataset; parallel evaluation and loss‑curve visualization.       |
| **10_ViT_Image_Classification.ipynb**   | Vision Transformer on Cats vs Dogs; mixed‑precision training, inference demo & performance metrics. |
| **11_Spam_Probing_Pretrained_LLMs.ipynb** | Frozen DistilBERT/TinyBERT embeddings + MLP head for Enron spam detection; accuracy, precision & recall analysis. |

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, CUDA, mixed‑precision  
- **NLP & Transformers:** Hugging Face Transformers, tokenizers  
- **Data & Analytics:** NumPy, Pandas, scikit‑learn  
- **Visualization & Logging:** Matplotlib, Seaborn, TensorBoard  

---

## ⚙️ Getting Started

```bash
git clone https://github.com/dhshah25/DeepLearning_Combined.git
cd DeepLearning_Combined
pip install -r requirements.txt
