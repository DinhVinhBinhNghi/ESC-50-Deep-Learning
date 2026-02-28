# Environmental Sound Classification (ESC) using Deep Learning
**Audio Processing · Feature Extraction · Convolutional Neural Networks (CNNs)**

This repository contains a deep learning research project focused on classifying environmental sounds using the ESC-50 dataset. The project demonstrates the end-to-end pipeline of audio data preprocessing, feature extraction, neural network modeling, and rigorous evaluation.

📄 **[Read the Full Research Paper Here](Nhận_Dạng_Âm_Thanh_Môi_Trường.pdf)** 
 
## 🎯 Objectives & Scope
* **Objective:** Accurately classify raw audio waveforms into 50 distinct environmental sound categories.
* **Dataset:** ESC-50 (2,000 labeled environmental audio recordings, 5 seconds each).
* **Research Focus:** Evaluate the effectiveness of different deep learning architectures in pattern recognition for audio signals.

## 🧠 Methodology & Architecture
* **Feature Extraction:** Transformed raw 1D audio signals into 2D representations (Mel-spectrograms and MFCCs) to capture temporal and spectral features.
* **Deep Learning Model:** Designed and trained Convolutional Neural Networks (CNNs) to recognize complex audio patterns.
* **Optimization:** Applied regularization techniques and hyperparameter tuning to prevent overfitting on a relatively small dataset.
* **Evaluation:** Utilized accuracy, precision, recall, and confusion matrices to critically evaluate model performance across different sound categories.

## 🚀 How to Run
This project was developed and executed using Google Colab to leverage cloud GPU acceleration.
1. Open the notebook directly in Colab: [![Open In Colab](https://colab.research.google.com/drive/1Se_QqukMkBvJwWNM8sLlHxy8agFjyqlD)]
2. Or clone this repository and run `ESC_50_Classification.ipynb` in your local Jupyter environment.

## 🧰 Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch / TensorFlow *(💡 Tip: Giữ lại framework bạn đã dùng, ưu tiên PyTorch vì JD yêu cầu)*
* **Audio Processing:** `librosa`
* **Data Science:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`
