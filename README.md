# Spam Email Detector using NLP techniques

A comprehensive spam detection application built with Python and Streamlit, featuring **different machine learning algorithms** for email classification. Compare algorithm performance, visualize results, and understand how different ML models detect spam!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## ✨ Features

- **8 Machine Learning Algorithms**: Compare Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Linear SVM, and K-Nearest Neighbors
- **Interactive Web Interface**: Built with Streamlit for easy testing and visualization
- **Real-time Classification**: Test any email text instantly with your chosen algorithm
- **Performance Comparison**: Side-by-side comparison of all algorithms with metrics
- **Visual Analytics**: 
  - Interactive charts with Plotly
  - Confusion matrices for each model
  - Feature importance visualization
  - Email characteristics analysis (length, word count, special characters)
- **Email Feature Analysis**: Understand what makes an email spam through data visualization
- **Pre-trained Models**: Cached models for instant predictions

## 🎥 Demo

**Live demo: https://emailspam-app.streamlit.app/**


## 🧠 Algorithms Included

### Simple Algorithms (Great for Learning)
1. **Logistic Regression** ⭐ - Fast, interpretable, excellent for text classification
2. **Naive Bayes** - Classic spam detection algorithm, extremely fast
3. **K-Nearest Neighbors** - Instance-based learning, simple concept

### Tree-Based Algorithms
4. **Decision Tree** - Visual, easy to interpret decision rules
5. **Random Forest** - Ensemble of 100 trees, robust and accurate
6. **Gradient Boosting** - Sequential learning from mistakes, powerful
7. **XGBoost** - State-of-the-art gradient boosting, Kaggle favorite

### Other Algorithms
8. **Linear SVM** - Excellent for high-dimensional text data


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/leonardstaniloiu/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```txt
pandas
numpy
streamlit
scikit-learn
xgboost
plotly
matplotlib
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```
OR
```bash
python -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`


## 🔬 How It Works

### 1. Data Loading & Preprocessing
- Load spam/ham emails from CSV
- Normalize column names
- Create binary labels (1=spam, 0=ham)
- Extract features: message length, word count, special characters

### 2. Text Vectorization (TF-IDF)
- Convert text to numerical features
- TF-IDF (Term Frequency-Inverse Document Frequency) weighting
- Capture unigrams and bigrams
- Filter stop words

### 3. Model Training
- Split data: 80% training, 20% testing
- Train all 8 algorithms on the same data
- Stratified split to maintain spam/ham ratio
- Cache trained models for performance

### 4. Prediction & Evaluation
- Predict on test set
- Calculate metrics: Accuracy, Precision, Recall, F1-Score
- Generate confusion matrices
- Provide probability scores where available

### Pipeline Visualization

```
Email Text
    ↓
TF-IDF Vectorization
    ↓
Numerical Features (3000-dimensional vector)
    ↓
ML Algorithm (8 options)
    ↓
Prediction: Spam (1) or Ham (0)
    ↓
Confidence Score (0-100%)
