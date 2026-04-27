import numpy as np
import tensorflow as tf
from pydub import AudioSegment

# Function to generate sound wave for a note
def generate_note_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration/1000, int(sample_rate*duration / 1000), False)
    wave = 0.5 * np.sin(2*np.pi*freq*t)  # volume adjusted to avoid clipping
    return (wave*32767).astype(np.int16)  # Convert to 16-bit data

# Create note audio
def create_note_sound(note_name, duration):
    wave_data = generate_note_wave(notes_freq[note_name], duration, sample_rate)
    return AudioSegment(wave_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

# Parameters
sample_rate = 44100  # samples per second

# Define note frequencies
notes_freq = {
    'A': 440.00,
    'C': 261.63,
    'E': 329.63,
    'F': 349.23,
    'G': 392.00,
    'kz1':14,
    'kz2':16,
    'kz3':18,
    'kz4':20,
    'kz5':22,
    'kz6':24,
    'kz7':26,
    'kz8':28,
    'kz9':30,
    'kz10':32,
    'kz11':36,
    'kz12':38,
    'kz13':40,
}

# Define melody sequence
# sequence = ['C','kz1','C','kz2','C','kz3','C','kz4','C','kz5','C','kz6','C','kz7']
sequence = ['kz1','kz2','kz3','kz4','kz5','kz6','kz7','kz8','kz9','kz10','kz11','kz12','kz13']

# Combine notes into song
song = AudioSegment.silent(duration=0)
for note in sequence:
    if note == "C":
        song += create_note_sound(note, duration=500)
    else:
        song += create_note_sound(note, duration=2000)

# Save to file
song.export("song_output.wav", format="wav")
