# 📡 SSTV Image Transmitter & Receiver (Python + Streamlit)

This project simulates **Slow Scan Television (SSTV)** to **encode images into audio** and **decode them back from audio** using only Python — with no hardware dependencies!

🚀 Made for college demonstration, hobby experiments, and pure curiosity about visual radio transmission.

---

## 🌟 Features

- ✅ Encode full-color images to SSTV `.wav` files (Martin M1 simulation)
- ✅ Decode `.wav` audio files back into color images
- ✅ NEW: 🎙️ Live microphone decoding — play audio from another device and decode in real time
- ✅ Streamlit-based GUI — beginner-friendly and easy to 
- ✅ Zero-cost, 100% Python project — no SDR hardware needed

---

## 🖼️ How SSTV Works (Simplified)

SSTV is a way to send images using sound. Each pixel's color is turned into a specific **frequency tone**, and these tones are transmitted line by line. The receiver listens to the tones and reconstructs the image.

This project simulates a **Martin M1-like** SSTV mode:
- Red, Green, Blue channels sent line-by-line
- Sync and separator tones simulated
- FFT used to decode frequency → pixel

---

## 📂 Folder Structure

```
sstv-python/
├── app.py                  # Main Streamlit App
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

---

## 🛠️ Installation

1. **Clone the repo**:
```bash
git clone https://github.com/ayushdubey570/sstv.git
cd sstv
```

2. **Install requirements**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
streamlit run app.py
```

---

## 🧪 Usage

### 🎨 Encode Image to SSTV Audio

1. Select `Encode Image to SSTV` in the sidebar.
2. Upload any `.jpg` or `.png` image.
3. Audio will be generated and playable.
4. Download `.wav` and test.

---

### 🔊 Decode SSTV from Audio File

1. Select `Decode SSTV from File`.
2. Upload any `.wav` file generated from encoder or external device.
3. Image will be decoded and displayed.

---

### 🎤 Live SSTV Decoder (Mic Input)

1. Select `Decode SSTV Live (Mic)`.
2. Click “Start Recording” and play SSTV sound from another device.
3. It will listen through your microphone and decode the image live.

---

## 📦 Requirements

- Python 3.8+
- Streamlit
- NumPy
- OpenCV
- SciPy
- sounddevice (for mic input)

Install them via:

```bash
pip install streamlit numpy opencv-python scipy sounddevice
```
Rn via Streamlit:

```bash
streamlit run app.py
```


---

## 📚 Learn More

- SSTV Wikipedia: https://en.wikipedia.org/wiki/Slow-scan_television
- Martin M1 format specs
- RTL-SDR and ham radio communities for real-world SSTV examples

---

## 🙌 Credits

Developed by [Ayush](https://github.com/ayushdubey570)

Inspired by the open-source SSTV protocols and amateur radio experiments.

---

## 📄 License

This project is open-source and free to use for educational or non-commercial use.
