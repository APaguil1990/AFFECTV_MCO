import joblib
import numpy as np
import re

from tensorflow.keras.models import load_model
model = load_model('emotion_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
import tkinter as tk
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

custom_stop_words = {
    "i", 
    "im", 
    "ive", 
    "dont", 
    "and", 
    "you", 
    "he", 
    "she", 
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

# Preprocess text (lowercase, remove stopwords, tokenize)
def preprocess_text(text): 

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words) 

class MyGUI():
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x500")
        self.root.title("AFFECTV MCO")
        
        self.label = tk.Label(self.root, text="Live Text-Emotion Recognition", font=('Helvetica', 20))
        self.label.pack(pady=20)

        self.textbox = tk.Text(self.root, height=2, font=('Arial', 25))
        self.textbox.pack(padx=10)

        self.buttonframe = tk.Frame(self.root)
        self.buttonframe.columnconfigure(0, weight=1)
        self.buttonframe.columnconfigure(1, weight=1)

        self.convertbutton = tk.Button(self.buttonframe, text="Emotion?", font=('Comic Sans', 20), command=self.create_label)
        self.convertbutton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.resetbutton = tk.Button(self.buttonframe, text="Reset", font=('Comic Sans', 20), command=self.clear_screen)
        self.resetbutton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.buttonframe.pack(fill='x')
                        

        self.root.mainloop()
        
    def create_label(self):
    
        input_text = self.textbox.get('1.0', tk.END).strip()  
        preprocessed_text = preprocess_text(input_text)  

        new_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()

        predicted_label = model.predict(new_vector)
        predicted_class = np.argmax(predicted_label, axis=1)  
        predicted_emotion = label_encoder.inverse_transform(predicted_class)  

        
        if hasattr(self, 'newlabel') and self.newlabel:
            self.newlabel.destroy()

        self.newlabel = tk.Label(self.root, text=f"Emotion: {predicted_emotion[0]}", font=('Arial', 30))
        self.newlabel.pack(pady=10)  # Add some pa
        
    def clear_screen(self):
        self.textbox.delete(1.0, tk.END)
        if hasattr(self, 'newlabel') and self.newlabel:
            self.newlabel.destroy()  
            
MyGUI()