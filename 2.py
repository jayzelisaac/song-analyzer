import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np

class Tonal_Fragment:
    def __init__(self, y_harmonic, sr):
        self.y_harmonic = y_harmonic
        self.sr = sr

    def print_key(self):
        # Extract harmonic pitch class profile (HPCP)
        chroma = librosa.feature.chroma_cqt(y=self.y_harmonic, sr=self.sr)
        
        # Compute the average chroma vector
        chroma_mean = np.mean(chroma, axis=1)

        # Use librosa's in-built method to estimate key
        key = librosa.core.pitch.chroma_to_key(chroma_mean)
        
        detected_key = key
        return detected_key, 1.0  # Assuming a default strength of 1.0 for simplicity

class KeyFinderApp:
    def __init__(self, root):
        self.root = root
        root.title("Musical Key Finder")

        self.label = tk.Label(root, text="Select a song to find its key", height=4)
        self.label.pack()

        self.select_button = tk.Button(root, text="Select Song", command=self.select_song)
        self.select_button.pack()

        self.result_label = tk.Label(root, text="", height=4)
        self.result_label.pack()

    def select_song(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            key, strength = self.analyze_song(file_path)
            self.result_label.config(text=f"Detected Key: {key} (Strength: {strength:.2f})")

    def analyze_song(self, file_path):
        y, sr = librosa.load(file_path)
        y_harmonic, _ = librosa.effects.hpss(y)
        song = Tonal_Fragment(y_harmonic, sr)
        key, strength = song.print_key()  # Assuming print_key returns a tuple (key, strength)
        return key, strength

if __name__ == "__main__":
    root = tk.Tk()
    app = KeyFinderApp(root)
    root.mainloop()
