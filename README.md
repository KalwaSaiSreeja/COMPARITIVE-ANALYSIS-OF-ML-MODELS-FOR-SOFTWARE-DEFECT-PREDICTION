
# 🧠 Comparative Analysis of Machine Learning Models for Software Defect Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![scikit-learn](https://img.shields.io/badge/ML-sklearn-orange.svg)](https://scikit-learn.org/)
[![WEKA](https://img.shields.io/badge/Tool-WEKA-red.svg)](https://www.cs.waikato.ac.nz/ml/weka/)

---

## 📌 Project Objectives

- Develop a Python GUI for software defect classification.
- Implement and evaluate Naive Bayes, Decision Tree, and Random Forest.
- Compare model performance with the WEKA tool.
- Analyze accuracy, F1-score, MAE, RMSE, and more.
- Identify the best-performing model for software defect prediction.

---

## 🛠️ Technologies Used

### Python-Based GUI
- **Language**: Python 3.x
- **Libraries**: 
  - `PyQt5` (GUI)
  - `pandas`, `numpy` (Data Handling)
  - `scikit-learn` (Machine Learning)

### Benchmark Tool
- **WEKA**: Java-based platform for data mining and ML analysis.

---

## 📁 Features

### ✅ Python GUI Highlights
- Interactive GUI with two-panel layout.
- Supports loading training/testing datasets (CSV).
- Options to check null values and class imbalance.
- Dropdown model selection: Naive Bayes, Decision Tree, Random Forest.
- Performance evaluation:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-score
  - MAE, RMSE, RAE, RRSE
  - Kappa Statistic

### 📉 WEKA Usage
- Dataset preprocessing, model training, and evaluation.
- Same set of models evaluated for cross-comparison.

---

## 🧪 Dataset Used

- **JM1 Dataset** from NASA's PROMISE repository.
- Dataset is split into train/validation/test (60/20/20).

---

## 📊 Results Summary

- **Random Forest** consistently outperformed others in both environments (Python & WEKA).
- Results vary across datasets; performance is model- and data-dependent.

---

## 🚧 Limitations

- Only one dataset tested.
- No hyperparameter tuning or cross-validation in Python GUI.
- Visual outputs are text-based only.

---

## 🔮 Future Scope

- Add more ML models (e.g., XGBoost, SVM, Neural Networks).
- Support hyperparameter tuning.
- Add graphical evaluation (ROC, Confusion Matrix Heatmaps).
- Include k-fold cross-validation.
- Package GUI as an executable or web tool.

---


## 👨‍💻 Authors

- **Kalwa Sai Sreeja** – Developer
- **Janani Prakash** – Co-Developer

Under the guidance of **Shri. Deepanshu Dixit**, DRDL Hyderabad

---

## 📄 License

This project is licensed under the **MIT License**.

MIT License

Copyright (c) 2025 Kalwa Sai Sreeja, Janani Prakash

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell   
copies of the Software, and to permit persons to whom the Software is        
furnished to do so, subject to the following conditions:                     

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.                              

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### 📘 Summary:
- ✅ Free to use, modify, and distribute.
- 🔗 Include credit to the authors.
- 🚫 No warranty or liability from the authors.




