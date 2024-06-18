import os
import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def analyze_audio(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Extract chroma feature
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Summing up chroma energies across all time frames
        chroma_energy = np.sum(chromagram, axis=1)
        
        # Determine the key
        key_index = np.argmax(chroma_energy)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        
        return key, tempo
    
    except Exception as e:
        messagebox.showerror("Error", f"Error processing {file_path}: {e}")
        return None, None

def select_file_and_analyze():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    
    if file_path:
        key, tempo = analyze_audio(file_path)
        
        if key and tempo:
            if isinstance(tempo, np.ndarray):
                tempo_str = ", ".join([f"{t:.2f} BPM" for t in tempo])
                messagebox.showinfo("Analysis Result", f"Key: {key}\nTempos: {tempo_str}")
            else:
                messagebox.showinfo("Analysis Result", f"Key: {key}\nTempo: {tempo:.2f} BPM")
        else:
            messagebox.showwarning("Analysis Result", "Unable to analyze the file.")

def create_gui():
    root = tk.Tk()
    root.title("MP3 Key & Tempo Analyzer")
    
    label = tk.Label(root, text="Select an MP3 file to analyze:")
    label.pack(pady=10)
    
    analyze_button = tk.Button(root, text="Select File", command=select_file_and_analyze)
    analyze_button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
