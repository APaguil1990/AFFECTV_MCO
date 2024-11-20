import streamlit as st
import pickle
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Load models and preprocessors
@st.cache_resource
def load_models():
    # Load Decision Tree Model
    with open("models/decision_tree_model.pkl", "rb") as f:
        decision_tree_model = pickle.load(f)
    
    # Load Neural Network Model
    neural_network_model = tf.keras.models.load_model("models/neural_network_model.h5")
    
    # Load TF-IDF Vectorizer
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Load Label Encoder
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load RoBERTa Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("models/roberta_tokenizer")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("models/roberta_model")

    return decision_tree_model, neural_network_model, tfidf_vectorizer, label_encoder, tokenizer, roberta_model

# Predict with Decision Tree
def predict_with_decision_tree(text, tfidf_vectorizer, decision_tree_model, label_encoder):
    vectorized_text = tfidf_vectorizer.transform([text])
    prediction = decision_tree_model.predict(vectorized_text)
    return label_encoder.inverse_transform(prediction)[0]

# Predict with Neural Network
def predict_with_neural_network(text, tfidf_vectorizer, neural_network_model, label_encoder):
    vectorized_text = tfidf_vectorizer.transform([text]).toarray()
    prediction = neural_network_model.predict(vectorized_text)
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)[0]

# Predict with RoBERTa
def predict_with_roberta(text, tokenizer, roberta_model):
    encoded_text = tokenizer(text, return_tensors="pt")
    output = roberta_model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    labels = ["Negative", "Neutral", "Positive"]
    return labels[scores.argmax()], scores

# Predict with VADER
def predict_with_vader(text, sia):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "Positive"
    elif scores["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title("Emotion Prediction App")

# Load models
decision_tree_model, neural_network_model, tfidf_vectorizer, label_encoder, tokenizer, roberta_model = load_models()
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Select a Machine Learning Model", ["Decision Tree", "Neural Network", "RoBERTa", "VADER"])

# Text Input
st.write("Enter the text to analyze:")
user_input = st.text_area("Your Input:")

# Analyze Button
if st.button("Analyze"):
    if user_input.strip():
        if model_choice == "Decision Tree":
            emotion = predict_with_decision_tree(user_input, tfidf_vectorizer, decision_tree_model, label_encoder)
            st.write(f"Predicted Emotion: **{emotion}**")
        elif model_choice == "Neural Network":
            emotion = predict_with_neural_network(user_input, tfidf_vectorizer, neural_network_model, label_encoder)
            st.write(f"Predicted Emotion: **{emotion}**")
        elif model_choice == "RoBERTa":
            emotion, scores = predict_with_roberta(user_input, tokenizer, roberta_model)
            st.write(f"Predicted Emotion: **{emotion}**")
            st.write(f"Confidence Scores: {scores}")
        elif model_choice == "VADER":
            emotion = predict_with_vader(user_input, sia)
            st.write(f"Predicted Emotion: **{emotion}**")
    else:
        st.warning("Please enter some text.")
