import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np

class KeyFinderApp:
    def __init__(self, master):
        self.master = master
        master.title("Key Finder")
        self.label = tk.Label(master, text="Select a song to find its key")
        self.label.pack()
        self.find_key_button = tk.Button(master, text="Find Key", command=self.select_song)
        self.find_key_button.pack()
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def select_song(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            key, strength = self.analyze_song(file_path)
            self.result_label.config(text=f"Detected Key: {key} (Strength: {strength:.2f})")

    def analyze_song(self, file_path):
        y, sr = librosa.load(file_path)
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        key, strength = self.detect_key(chroma)
        return key, strength

    def detect_key(self, chroma):
        # This is a placeholder for actual key detection logic.
        # You would analyze the chroma array to determine the key.
        # For simplicity, let's assume C Major for now.
        return "C Major", 1.0  # Placeholder values

if __name__ == "__main__":
    root = tk.Tk()
    app = KeyFinderApp(root)
    root.mainloop()