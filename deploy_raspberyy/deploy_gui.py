import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import numpy as np
import librosa
import tensorflow as tf



# Fungsi untuk merekam audio
def record_audio():
    try:
        command = [
            "arecord",
            "--format=S16_LE",
            "--duration=3",
            "--rate=16000",
            "--file-type=wav",
            "sample.wav"
        ]
        subprocess.run(command, check=True)
        messagebox.showinfo("Info", "Rekaman selesai! File disimpan sebagai sample.wav.")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat merekam: {e}")


# Fungsi untuk memutar audio
def play_audio():
    try:
        command = [
            "aplay",
            "--format=S16_LE",
            "--rate=16000",
            "sample.wav"
        ]
        subprocess.run(command, check=True)
        messagebox.showinfo("Info", "Memutar audio selesai!")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat memutar audio: {e}")


def classify_audio():
    audio_file_path = "sample.wav"  # File audio hasil rekaman
    tflite_model_path = "ann.tflite"  # Path ke model TFLite

    try:
        # Load and prepare the audio file
        y, sr = librosa.load(audio_file_path, sr=16000)  # Load audio file

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        rmse = librosa.feature.rms(y=y).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3).mean(axis=1)[:3]

        # Combine all features into a single array (20 features total)
        features = np.hstack([mfcc, zcr, spectral_centroid, spectral_bandwidth, rmse, spectral_contrast])
        features = np.expand_dims(features, axis=0).astype(np.float32)

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], features)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data, axis=1)[0]

        # Label mapping
        label_order = ["neutral", "low stress", "high stress"]
        predicted_label = label_order[predicted_class_index]
        confidence = output_data[0][predicted_class_index] * 100

        # Display the result
        messagebox.showinfo("Klasifikasi", f"Hasil Prediksi: {predicted_label}\nKepercayaan: {confidence:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat klasifikasi: {e}")



# GUI dengan Tkinter
def create_gui():
    root = tk.Tk()
    root.title("Audio Recorder & Classifier")

    # Button untuk merekam audio
    record_button = tk.Button(root, text="Record", command=record_audio, width=20, height=2)
    record_button.pack(pady=10)

    # Button untuk memutar audio
    play_button = tk.Button(root, text="Play", command=play_audio, width=20, height=2)
    play_button.pack(pady=10)

    # Button untuk klasifikasi
    classify_button = tk.Button(root, text="Classify", command=classify_audio, width=20, height=2)
    classify_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
