import streamlit as st
import numpy as np
#import cv2 - it is not supported by streamlit cloud, you can keep it while running locally
import scipy.io.wavfile as wav
import tempfile
import os
import sounddevice as sd
import queue
import threading
from io import BytesIO

# Constants for RGB SSTV simulation (Martin M1-like)
SAMPLE_RATE = 44100
LINE_DURATION = 0.275  # seconds per color channel
IMG_WIDTH = 320
IMG_HEIGHT = 256
FREQ_SYNC = 1200
FREQ_SEP = 1500
FREQ_LOW = 1500
FREQ_HIGH = 2300


def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def image_to_sstv_wav(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    audio = []

    for y in range(IMG_HEIGHT):
        row = img[y]

        # Sync pulse
        audio.extend(generate_tone(FREQ_SYNC, 0.01))
        # Separator tone
        audio.extend(generate_tone(FREQ_SEP, 0.003))

        for channel in range(3):  # R, G, B
            for x in range(IMG_WIDTH):
                pixel_value = row[x][channel]
                freq = np.interp(pixel_value, [0, 255], [FREQ_LOW, FREQ_HIGH])
                audio.extend(generate_tone(freq, LINE_DURATION / IMG_WIDTH))

    audio = np.array(audio)
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, SAMPLE_RATE, scaled)
    return temp_wav.name


def sstv_wav_to_image(wav_path):
    rate, data = wav.read(wav_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    samples_per_pixel = int(rate * LINE_DURATION / IMG_WIDTH)
    samples_per_line = samples_per_pixel * IMG_WIDTH
    samples_per_rgb = samples_per_line * 3 + int(rate * 0.013)  # Sync + Sep + 3 channels

    total_lines = len(data) // samples_per_rgb
    image = np.zeros((total_lines, IMG_WIDTH, 3), dtype=np.uint8)

    for i in range(total_lines):
        offset = i * samples_per_rgb + int(rate * 0.013)  # skip sync + sep
        for c in range(3):
            for j in range(IMG_WIDTH):
                idx = offset + (c * samples_per_line) + (j * samples_per_pixel)
                segment = data[idx:idx + samples_per_pixel]
                if len(segment) < samples_per_pixel:
                    continue
                fft = np.fft.fft(segment * np.hamming(len(segment)))
                freq = np.argmax(np.abs(fft[:samples_per_pixel // 2])) * rate / samples_per_pixel
                pixel = int(np.clip(np.interp(freq, [FREQ_LOW, FREQ_HIGH], [0, 255]), 0, 255))
                image[i, j, c] = pixel

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_img.name, image)
    return temp_img.name


# Real-time audio capture from microphone
audio_queue = queue.Queue()
recording_flag = threading.Event()


def audio_callback(indata, frames, time, status):
    if recording_flag.is_set():
        audio_queue.put(indata.copy())


def record_audio(duration=10):
    recording_flag.set()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        sd.sleep(int(duration * 1000))
    recording_flag.clear()

    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())

    audio_data = np.concatenate(frames, axis=0)
    scaled = np.int16(audio_data.flatten() * 32767)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, SAMPLE_RATE, scaled)
    return temp_wav.name


# Streamlit GUI
st.title("📡 SSTV (Slow Scan TV) - Decode your Audio file back into Image")
st.markdown("Transmit and Receive Color Images using Simulated SSTV (Martin M1-like)")

option = st.radio("Choose Mode", ["Encode Image to SSTV", "Decode SSTV from File", "Decode SSTV Live (Mic)"])

if option == "Encode Image to SSTV":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(uploaded_image.read())
            wav_path = image_to_sstv_wav(temp.name)

        st.audio(wav_path)
        st.success("Color SSTV Audio Signal Generated!")
        with open(wav_path, "rb") as file:
            st.download_button("Download SSTV .wav", file, file_name="sstv_color_output.wav")

#elif option == "Decode SSTV from File":
#    uploaded_audio = st.file_uploader("Upload SSTV .wav File", type=["wav"])
#    if uploaded_audio is not None:
#        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
#            temp.write(uploaded_audio.read())
#            img_path = sstv_wav_to_image(temp.name)
#keep these files while running locally-----> as it is not supported by streamlit cloud
        st.image(img_path, caption="Decoded Color Image")
        st.success("Color Image Decoded from SSTV Audio!")
        with open(img_path, "rb") as file:
            st.download_button("Download Decoded Image", file, file_name="decoded_color_image.png")
