import tkinter as tk

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
    # Create a new label and pack it at the bottom
    
        if hasattr(self, 'newlabel') and self.newlabel:
            self.newlabel.destroy()
     
        self.newlabel = tk.Label(self.root, text=self.textbox.get('1.0', tk.END), font=('Arial', 30))
        self.newlabel.pack(pady=10)  # Add some padding for spacekk
        
    def clear_screen(self):
        self.textbox.delete(1.0, tk.END)
        if hasattr(self, 'newlabel') and self.newlabel:
            self.newlabel.destroy()  
            
MyGUI()