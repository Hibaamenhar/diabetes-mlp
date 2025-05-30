
# Diabetes MLP Classifier (Pima Indians Dataset)

This project implements a Multilayer Perceptron (MLP) from scratch using only NumPy to perform binary classification on the Pima Indians Diabetes dataset.

## 🧠 Objective

To understand and build a neural network manually without using deep learning frameworks. This includes:

- Data preprocessing
- Forward and backward propagation
- Loss computation (Binary Cross-Entropy)
- Mini-batch Stochastic Gradient Descent (SGD)
- Metrics evaluation (Accuracy, F1-score, etc.)

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: 8 numeric medical predictor variables
- Target: Binary classification (0: non-diabetic, 1: diabetic)

## 🛠️ Project Structure

```
diabetes_mlp_project/
├── code/
│   ├── main.py                  # Main script to load, train and evaluate the model
│   └── neural_network.py        # Neural network class definition
├── figures/
│   ├── loss_curve.png           # Training loss plot
│   └── confusion_matrix.png     # Confusion matrix from test evaluation
├── diabetes_article.tex         # LaTeX report following IMRAD structure
```

## 🚀 How to Run

1. Clone this repository and navigate to the `code/` directory.
2. Make sure you have Python 3.7+ installed and run:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   python main.py
   ```
3. Results will be saved in the `figures/` directory.

## 📄 Scientific Report

You can find the full report under `diabetes_article.tex` (IMRAD structure). Compile it using:
```bash
pdflatex diabetes_article.tex
```

Or upload it to [Overleaf](https://overleaf.com) for online editing and PDF generation.

## 📌 Author

Master Student – IMSD / IAA – Academic Year 2024–2025
