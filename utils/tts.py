import pyttsx3
import os

def generate_audio(text, output_path):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file(text, output_path)
    engine.runAndWait()