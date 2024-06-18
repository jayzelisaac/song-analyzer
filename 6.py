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
            key = self.analyze_song(file_path)
            self.result_label.config(text=f"Detected Key: {key}")

    def analyze_song(self, file_path):
        # Load audio file and extract harmonic component
        y, sr = librosa.load(file_path)
        y_harmonic, _ = librosa.effects.hpss(y)

        # Extract chroma features using Librosa
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # Compute mean of chroma features
        chroma_mean = np.mean(chroma, axis=1)

        # Define Krumhansl-Schmuckler key profiles for major and minor keys
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Initialize variables for key detection
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

            # Check if the correlation is higher than the current maximum
            if correlation_major > max_correlation:
                max_correlation = correlation_major
                detected_key = f"{librosa.midi_to_note(60 + i)} Major"

            if correlation_minor > max_correlation:
                max_correlation = correlation_minor
                detected_key = f"{librosa.midi_to_note(60 + i)} Minor"

        return detected_key

if __name__ == "__main__":
    root = tk.Tk()
    app = KeyFinderApp(root)
    root.mainloop()
