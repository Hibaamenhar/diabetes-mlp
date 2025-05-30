
# Diabetes MLP Classifier

This project implements a Multilayer Perceptron (MLP) from scratch using NumPy to perform binary classification on the Pima Indians Diabetes dataset.

## 📚 Description

The MLP is trained to classify whether a person is diabetic or not, based on medical features from the dataset. The implementation includes:

- Manual forward and backward propagation
- Binary Cross-Entropy loss
- Training using mini-batch Stochastic Gradient Descent (SGD)
- No use of machine learning frameworks like TensorFlow or PyTorch

## 📁 Project Structure

```
diabetes_mlp_project/
├── code/
│   ├── main.py                # Runs the model training and evaluation
│   ├── neural_network.py      # Contains the MLP implementation
│   └── diabetes.csv           # Dataset (from Kaggle)
├── figures/
│   ├── loss_curve.png         # Training loss curve
│   └── confusion_matrix.png   # Evaluation confusion matrix
├── diabetes_article.tex       # Scientific report in LaTeX (IMRAD structure)
├── README.md                  # This file
```

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 samples with 8 features and 1 binary target (`Outcome`)

## 🚀 How to Run

1. Make sure you have Python 3 installed
2. Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

3. Run the training script:

```bash
cd code
python3 main.py
```

The script will:
- Train the MLP
- Output classification metrics
- Save figures into the `figures/` folder

## 📄 Report

A LaTeX report is included (`diabetes_article.tex`) and can be compiled with Overleaf or `pdflatex`.

## 🔗 GitHub

Project repository: [https://github.com/Hibaamenhar/diabetes-mlp](https://github.com/Hibaamenhar/diabetes-mlp)

## 👤 Author

Master Student – IAA  
Academic Year 2024–2025
