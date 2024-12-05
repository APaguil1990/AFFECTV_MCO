import re 
import tkinter as tk 
from tkinter import ttk 
import numpy as np 
import joblib 
from tensorflow.keras.models import load_model 
import pickle 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from sklearn.tree import DecisionTreeClassifier

# Load the models and vectorizers
nn_model = load_model(r"C:\AFFECTV\AFFECTV_MCO\emotion_model_combined.h5")
dt_model = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\decision_tree_model_with_features.joblib")
label_encoder = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\label_encoder_combined.pkl")
tfidf_vectorizer_text = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\tfidf_vectorizer_text.pkl")
tfidf_vectorizer_processed = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\tfidf_vectorizer_processed.pkl")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

custom_stop_words = {"and", "we", "they", "it", "is", "are", "was", "were", "their", "there"}
stop_words.update(custom_stop_words)

def preprocess_text(text):
    # Convert to lowercase, remove special characters, and tokenize
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

class EmotionPredictionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x500")
        self.root.title("Emotion Recognition GUI")
        self.root.configure(bg="black")

        # Title
        self.label = tk.Label(self.root, text="Text Emotion Recognition", font=("Helvetica", 20), fg="white", bg="black")
        self.label.pack(pady=20)

        # Text input box
        self.textbox = tk.Text(self.root, height=2, font=("Arial", 25), bg="gray10", fg="white")
        self.textbox.pack(padx=10)

        # Dropdown for model selection
        self.dropdown_label = tk.Label(self.root, text="Select Model:", font=("Arial", 15), fg="white", bg="black")
        self.dropdown_label.pack(pady=10)
        self.model_selector = tk.StringVar(value="Neural Network")
        self.dropdown = tk.OptionMenu(self.root, self.model_selector, "Neural Network", "Decision Tree")
        self.dropdown.config(font=("Arial", 12), bg="gray10", fg="white")
        self.dropdown["menu"].config(font=("Arial", 12), bg="gray10", fg="white")
        self.dropdown.pack(pady=10)

        # Buttons
        self.buttonframe = tk.Frame(self.root, bg="black")
        self.buttonframe.columnconfigure(0, weight=1)
        self.buttonframe.columnconfigure(1, weight=1)
        self.predict_button = tk.Button(self.buttonframe, text="Predict Emotion", font=("Comic Sans", 20),
                                        command=self.predict_emotion, bg="gray20", fg="white")
        self.predict_button.grid(row=0, column=0, sticky=tk.W + tk.E)
        self.reset_button = tk.Button(self.buttonframe, text="Reset", font=("Comic Sans", 20),
                                       command=self.clear_screen, bg="gray20", fg="white")
        self.reset_button.grid(row=0, column=1, sticky=tk.W + tk.E)
        self.buttonframe.pack(fill="x")

        # Result display
        self.result_label = tk.Label(self.root, text="", font=("Arial", 25), fg="white", bg="black")
        self.result_label.pack(pady=10)

        self.root.mainloop()

    def predict_emotion(self):
        input_text = self.textbox.get("1.0", tk.END).strip()
        if not input_text:
            self.result_label.config(text="Please enter some text.", fg="red")
            return

        # Preprocess and generate features
        preprocessed_text = tfidf_vectorizer_text.transform([input_text]).toarray()
        preprocessed_processed = tfidf_vectorizer_processed.transform([preprocess_text(input_text)]).toarray()
        numerical_features = np.array([[len(input_text), len(input_text.split())]])
        new_features = np.hstack([preprocessed_text, preprocessed_processed, numerical_features])

        # Match input shape with the training data
        # Match input shape with the training data
        if new_features.shape[1] > dt_model.n_features_in_:
            new_features = new_features[:, :dt_model.n_features_in_]
        elif new_features.shape[1] < dt_model.n_features_in_:
            padding = np.zeros((new_features.shape[0], dt_model.n_features_in_ - new_features.shape[1]))
            new_features = np.hstack([new_features, padding])

        # Model prediction
        selected_model = self.model_selector.get()
        if selected_model == "Neural Network":
            predicted_label = nn_model.predict(new_features)
            predicted_class = np.argmax(predicted_label, axis=1)
        elif selected_model == "Decision Tree":
            predicted_class = dt_model.predict(new_features)
        else:
            self.result_label.config(text="Invalid model selected.", fg="red")
            return

        predicted_emotion = label_encoder.inverse_transform(predicted_class)
        self.result_label.config(text=f"The emotion of the text is: {predicted_emotion[0]}", fg="white")

    def clear_screen(self):
        self.textbox.delete("1.0", tk.END)
        self.result_label.config(text="")

EmotionPredictionGUI()