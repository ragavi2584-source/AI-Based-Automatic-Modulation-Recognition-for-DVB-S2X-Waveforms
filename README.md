

#AI-Based Automatic Modulation Recognition (AMR) for DVB-S2X Waveforms

Overview

This project is an AI-powered Automatic Modulation Recognition (AMR) system that identifies DVB-S2X modulation schemes (16APSK, 32APSK, and 64APSK) from complex I/Q signal samples. It combines Digital Signal Processing (DSP) and a Convolutional Neural Network (CNN) to provide accurate modulation classification and visualization.

Features

- Automatic classification of 16APSK, 32APSK, and 64APSK.
- Deep learning model using TensorFlow/Keras.
- Signal preprocessing and normalization.
- Constellation, waveform, FFT, and spectrogram visualization.
- Flask-based web interface for uploading I/Q signals.
- Confidence score and prediction time.
- Modular and extensible architecture.

Project Structure

AMR_DVB_S2X/
│
├── app.py
├── model.py
├── train.py
├── predict.py
├── preprocess.py
├── generate_dataset.py
├── visualization.py
├── requirements.txt
│
├── dataset/
├── saved_model/
├── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   ├── app.js
│   └── plots/
│
└── README.md

Installation

1. Clone the repository:

git clone <repository-url>
cd AMR_DVB_S2X

2. Install dependencies:

pip install -r requirements.txt

3. Generate the dataset:

python generate_dataset.py

4. Preprocess the dataset:

python preprocess.py

5. Train the CNN model:

python train.py

6. Start the Flask application:

python app.py

Usage

1. Open your browser and visit:

http://127.0.0.1:5000

2. Upload a ".npy" I/Q signal file.

3. Click Predict Modulation.

4. View:

- Predicted modulation type.
- Confidence score.
- Prediction time.
- Constellation diagram.
- Waveform.
- FFT spectrum.
- Spectrogram.

Technologies Used

- Python
- TensorFlow
- Keras
- Flask
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

Applications

- Satellite communication monitoring.
- Software Defined Radio (SDR).
- Cognitive radio.
- Signal intelligence.
- Wireless communication research.
- Academic demonstrations.

Future Enhancements

- Real-time SDR integration (RTL-SDR/HackRF).
- Support for additional modulation schemes.
- LSTM/Transformer-based models.
- Live streaming signal analysis.
- Docker deployment.
- REST API support.

License

This project is intended for educational, research, and hackathon purposes.