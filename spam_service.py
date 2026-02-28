import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

@st.cache_data
def load_data(path: str):
    try:
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

    df.columns = [col.strip().lower() for col in df.columns]

    if "category" not in df.columns or "message" not in df.columns:
        st.error("Dataset-ul trebuie să conțină coloanele 'category' și 'message'!")
        return pd.DataFrame()

    df = df[["category", "message"]].dropna()
    
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    
    df["label"] = (df["category"] == "spam").astype(int)

    df["message_length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    df["exclamation_count"] = df["message"].str.count("!")
    df["question_count"] = df["message"].str.count(r"\?")

    return df

def generate_random_email():
    emails = [
        "WINNER!! You have won a $100 gift card. Click here NOW to claim your prize!!!",
        "Congratulations! You've been selected for a free cruise to Barcelona! Call now to claim!!!",
        "URGENT!!! Your account has been compromised. Reset your password immediately by clicking this link.",
        "Get rich quick with this one simple trick! Click here to find out how!!!",
        "You have a new voicemail from your bank. Click here to listen to the message.",
        "Don't miss out on this limited time offer! Buy now and save 50%!!!",
        "Your package is waiting for you!!! Click here to schedule delivery.",
        "CONGRATULATIONS! You've won $500 in our lottery! Claim your prize now!!!",
        "ALERT! Your PayPal account needs verification. Click here to secure your funds!!!",
        "FREE IPHONE! Enter our contest and win an iPhone 15! Limited time offer!!!",
        "URGENT: Your Amazon order is on hold. Confirm delivery now!!!",
        "WIN BIG! Play our slots and win thousands! Click to start!!!",
        "EXCLUSIVE DEAL: 90% off all products! Shop now before it's gone!!!",
        "IMPORTANT: Update your bank details immediately to avoid account suspension!!!",
        "YOU'VE BEEN CHOSEN: Free vacation to Hawaii! Book now!!!",
        "MEGA WINNER! $1000 cash prize waiting for you! Click here to collect!!!",
        "HOT DEAL! Luxury watch for only $10! Order now before stock runs out!!!",
        "SECURITY ALERT: Your email account will be suspended. Verify now!!!",
        "INSTANT CASH: Make $1000 per day from home! Learn the secret!!!",
        "LAST CHANCE: Win a brand new car! Enter contest today!!!"
    ]
    return np.random.choice(emails)

def get_all_models():
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000,
                n_jobs=-1)
        },
        "Naive Bayes": {"model": MultinomialNB(alpha=1.0)},
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=10,
                random_state=42)
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        },
        "Linear SVM": {
            "model": LinearSVC(
                max_iter=1000, 
                random_state=42
                )
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
                )
        },
        "Simple Neural Network": {
            "model": MLPClassifier(
                hidden_layer_sizes=(100, 50),  
                activation='relu',  
                max_iter=500,
                learning_rate_init=0.001
            )
        },
    }
    return models

@st.cache_resource
def train_all_models(data_path: str):
    df = load_data(data_path)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    all_models = get_all_models()
    results = {}
    training_times = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model_info) in enumerate(all_models.items()):
        start_time = datetime.now()
        model = model_info["model"]
        model.fit(X_train_tfidf, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        training_times[name] = train_time

        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "train_time": train_time,
            "info": model_info,
        }

        progress_bar.progress((idx + 1) / len(all_models))

    progress_bar.empty()
    status_text.empty()

    return vectorizer, results, df, X_test, y_test


def predict_email(text: str, vectorizer, model):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    probability = None
    try:
        probability = model.predict_proba(text_tfidf)[0][1]
    except AttributeError:
        try:
            decision = model.decision_function(text_tfidf)[0]
            probability = 1 / (1 + np.exp(-decision))
        except Exception:
            pass

    return prediction, probability
