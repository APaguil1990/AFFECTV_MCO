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

nn_model = load_model(r"C:\AFFECTV\AFFECTV_MCO\emotion_model_combined.h5") 
dt_model = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\decision_tree_model_with_features.joblib") 
label_encoder = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\label_encoder_combined.pkl") 
tfidf_vectorizer = joblib.load(r"C:\AFFECTV\AFFECTV_MCO\tfidf_vectorizer_combined.pkl") 

neutral_sentiment = np.array( [[0, 0.5, 0.5, 0, 0.5, 0.5, 0.5]] ) 

nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

custom_stop_words = {
    "and", 
    "we", 
    "they", 
    "it", 
    "is", 
    "are", 
    "was", 
    "were", 
    "their", 
    "there"
}

stop_words.update(custom_stop_words) 

def preprocess_text(text): 
    # Convert lowercase 
    text = text.lower()
    # Remove special characters and numbers 
    text = re.sub(r'\W+', ' ', text) 
    # Tokenize 
    words = text.split() 
    # Remove stopwords 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words) 

class MyGUI: 
    def __init__(self): 
        self.root = tk.Tk() 
        self.root.geometry("800x500") 
        self.root.title("AFFECTV MCO GUI") 
        self.root.configure(bg="black") 

        self.label = tk.Label(
            self.root, 
            text = "Live Text-Emotion Recognition", 
            font = ("Helvetica", 20), 
            fg = "white", 
            bg = "black"
        )
        self.label.pack(pady=20)

        self.textbox = tk.Text(
            self.root, 
            height = 2, 
            font = ("Arial", 25), 
            bg = "gray10", 
            fg = "white"
        )
        self.textbox.pack(padx=10) 

        self.dropdown_label = tk.Label(
            self.root, 
            text = "Select Model:", 
            font = ("Arial", 15), 
            fg = "white", 
            bg = "black"
        ) 
        self.dropdown_label.pack(pady=10) 

        self.model_selector = tk.StringVar(value = "Neural Network") 
        
        self.dropdown = tk.OptionMenu(
            self.root, 
            self.model_selector, 
            "Neural Network", 
            "Decision Tree"
        ) 
        self.dropdown.config( font=("Arial", 12), bg="gray10", fg="white" ) 
        self.dropdown["menu"].config( font=("Arial", 12), bg="gray10", fg="white" ) 
        self.dropdown.pack(pady=10) 

        self.buttonframe = tk.Frame(self.root, bg="black") 
        self.buttonframe.columnconfigure(0, weight=1) 
        self.buttonframe.columnconfigure(1, weight=1) 

        self.predict_button = tk.Button(
            self.buttonframe, 
            text = "Predict Emotion", 
            font = ("Comic Sans", 20), 
            command = self.create_label, 
            bg = "gray20", 
            fg = "white"
        ) 
        self.predict_button.grid(row=0, column=0, sticky=tk.W + tk.E)

        self.reset_button = tk.Button(
            self.buttonframe, 
            text = "Reset", 
            font = ("Comic Sans", 20), 
            command = self.clear_screen, 
            bg = "gray20", 
            fg = "white"
        )
        self.reset_button.grid(row=0, column=1, sticky=tk.W + tk.E) 

        self.buttonframe.pack(fill="x") 

        self.result_label = tk.Label(
            self.root, 
            text = "", 
            font = ("Arial", 25), 
            fg = "white", 
            bg = "black"
        )
        self.result_label.pack(pady=10) 
        self.root.mainloop()

    def create_label(self): 
        input_text = self.textbox.get("1.0", tk.END).strip() 
        if not input_text: 
            self.result_label.config(text="Please enter some text.", fg="red") 
            return 
            
        preprocessed_text = preprocess_text(input_text) 
        tfidf_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray() 
        input_features = np.hstack([tfidf_vector, neutral_sentiment]) 

        selected_model = self.model_selector.get() 
        if selected_model == "Neural Network": 
            predicted_label = nn_model.predict(input_features) 
            predicted_class = np.argmax(predicted_label, axis=1) 
        elif selected_model == "Decision Tree": 
            predicted_class = dt_model.predict(input_features) 
        else: 
            self.result_label.config(text="Invalid model selected.", fg="red") 
            return 

        predicted_emotion = label_encoder.inverse_transform(predicted_class) 

        self.result_label.config(
            text = f"The emotion of the text is: {predicted_emotion[0]}", 
            fg = "white"
        )
    
    def clear_screen(self): 
        self.textbox.delete("1.0", tk.END) 
        self.result_label.config(text="") 


MyGUI()