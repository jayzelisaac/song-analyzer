import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np

class Tonal_Fragment:
    def __init__(self, y_harmonic, sr):
        self.y_harmonic = y_harmonic
        self.sr = sr

    def print_key(self):
        # Calculate the chroma feature
        chroma = librosa.feature.chroma_cqt(y=self.y_harmonic, sr=self.sr)
        print("Chroma Shape:", chroma.shape)
        
        # Compute the average chroma vector
        chroma_mean = np.mean(chroma, axis=1)
        print("Chroma Mean:", chroma_mean)
        
        # Define Krumhansl-Schmuckler key profiles for major and minor keys
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Initialize key and correlation values
        max_correlation = -1
        detected_key = None

        # Iterate over all 12 possible keys (12 major and 12 minor)
        for i in range(12):
            rotated_major_profile = np.roll(major_profile, i)
            rotated_minor_profile = np.roll(minor_profile, i)
            
            # Calculate correlation for major key
            correlation_major = np.corrcoef(chroma_mean, rotated_major_profile)[0, 1]
            # Calculate correlation for minor key
            correlation_minor = np.corrcoef(chroma_mean, rotated_minor_profile)[0, 1]
            
            # Debugging prints
            print(f"Key {librosa.midi_to_note(60 + i)} Major: {correlation_major:.2f}")
            print(f"Key {librosa.midi_to_note(60 + i)} Minor: {correlation_minor:.2f}")

            # Check if the correlation is higher than the current maximum
            if correlation_major > max_correlation:
                max_correlation = correlation_major
                detected_key = f"{librosa.midi_to_note(60 + i)} Major"
            
            if correlation_minor > max_correlation:
                max_correlation = correlation_minor
                detected_key = f"{librosa.midi_to_note(60 + i)} Minor"

        return detected_key, max_correlation

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
            key, correlation = self.analyze_song(file_path)
            self.result_label.config(text=f"Detected Key: {key} (Correlation: {correlation:.2f})")

    def analyze_song(self, file_path):
        y, sr = librosa.load(file_path)
        y_harmonic, _ = librosa.effects.hpss(y)
        song = Tonal_Fragment(y_harmonic, sr)
        key, correlation = song.print_key()  # Assuming print_key returns a tuple (key, correlation)
        return key, correlation

if __name__ == "__main__":
    root = tk.Tk()
    app = KeyFinderApp(root)
    root.mainloop()
