import sounddevice as sd
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
import time
import paho.mqtt.client as mqtt
import os
import wave
import tempfile
from scipy.fft import fft

# Configuration MQTT
MQTT_BROKER = "192.168.1.34"
MQTT_PORT = 1883
MQTT_USER = "Christian1"
MQTT_PASSWORD = "shbotBP"
MQTT_TOPIC = "home/sonnette"

# Configuration audio
CHUNK = 11025  # 0.5 seconde d'audio à 44100 Hz
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 2  # Index correspondant au ReSpeaker 4-Mic Array

# Paramètres de détection
DETECTION_THRESHOLD = 0.3
FREQ_THRESHOLD = 1270  # Fréquence principale de la sonnette
FREQ_TOLERANCE = 200    # Tolérance en Hz autour de la fréquence principale
MIN_AUDIO_POWER = 0.004
PROB_WINDOW = 10
MIN_DETECTION_DURATION = 0.2
MAX_DETECTION_DURATION = 3.0
CONSECUTIVE_DETECTIONS = 2

# Chemin vers le modèle
MODEL_PATH = "/home/pi/pyAudioAnalysis"
MODEL_NAME = "sonnette_model"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)

def notify_home_assistant(client, state, probability):
    message = f'{{"state": "{state}"}}'
    client.publish(MQTT_TOPIC, message)
    print(f"Notification envoyée à Home Assistant via MQTT: Sonnette {state} à {time.strftime('%Y-%m-%d %H:%M:%S')} (Probabilité: {probability:.2f})")

# Connexion MQTT
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

last_detection_time = 0
is_detecting = False
detection_start_time = 0

def classify_audio(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        try:
            result = aT.file_classification(temp_file.name, MODEL_FILE, "svm")
            probability = result[1][0]  # Probabilité que ce soit une sonnette
            print(f"Probabilité: {probability:.2f}")
            return probability
        except Exception as e:
            print(f"Erreur lors de la classification: {e}")
            return 0
        finally:
            os.unlink(temp_file.name)

def analyze_frequency(audio_data):
    freqs = fft(audio_data)
    freqs = np.abs(freqs[:len(freqs)//2])
    freq_values = np.fft.fftfreq(len(audio_data), 1/RATE)[:len(freqs)]
    peak_freq = freq_values[np.argmax(freqs)]
    print(f"Fréquence de pic: {peak_freq:.2f} Hz")
    return peak_freq

FREQ_WEIGHT = 0.5  # Poids de la correspondance de fréquence
PROB_WEIGHT = 0.5  # Poids de la probabilité du modèle
FREQ_RANGE_LOW = 1200  # Limite basse de la plage de fréquences (en Hz)
FREQ_RANGE_HIGH = 3300  # Limite haute de la plage de fréquences (en Hz)

FUNDAMENTAL_FREQ = 1270
SECOND_HARMONIC = 2530
THIRD_HARMONIC = 3266
FREQ_TOLERANCE = 100
MIN_PROB_THRESHOLD = 0.2
MIN_POWER_THRESHOLD = 0.03

prob_buffer = []
is_detecting = False
detection_start_time = None
consecutive_count = 0
last_detection_time = 0

def is_harmonic(freq, base_freq, tolerance):
    for n in range(1, 4):  # Vérifier jusqu'à la 3ème harmonique
        if abs(freq - (base_freq * n)) <= tolerance:
            return True
    return False

DETECTION_WINDOW = 4  # Nombre d'échantillons à considérer
detection_buffer = []

def is_doorbell_frequency(freq):
    harmonics = [FUNDAMENTAL_FREQ, SECOND_HARMONIC, THIRD_HARMONIC]
    return any(abs(freq - h) <= FREQ_TOLERANCE for h in harmonics)

def is_doorbell_sound(frequency, probability, power):
    return (probability >= MIN_PROB_THRESHOLD and 
            power >= MIN_POWER_THRESHOLD and 
            is_doorbell_frequency(frequency))

def audio_callback(indata, frames, time_info, status):
    global detection_buffer, is_detecting, detection_start_time

    audio = indata[:, 0]
    audio_power = np.mean(np.abs(audio))
    print(f"Puissance audio: {audio_power:.4f}")

    if audio_power >= MIN_POWER_THRESHOLD:
        probability = classify_audio(audio)
        peak_freq = analyze_frequency(audio)
        print(f"Probabilité: {probability:.2f}, Fréquence de pic: {peak_freq:.2f} Hz")

        detection_buffer.append((peak_freq, probability, audio_power))
        if len(detection_buffer) > DETECTION_WINDOW:
            detection_buffer.pop(0)

        if any(is_doorbell_sound(f, p, pw) for f, p, pw in detection_buffer):
            if not is_detecting:
                is_detecting = True
                detection_start_time = time.time()
                notify_home_assistant(mqtt_client, "on", max(p for _, p, _ in detection_buffer))
        elif is_detecting:
            detection_duration = time.time() - detection_start_time
            if detection_duration >= MIN_DETECTION_DURATION:
                is_detecting = False
                notify_home_assistant(mqtt_client, "off", max(p for _, p, _ in detection_buffer))

    else:
        print("Puissance audio insuffisante")
        detection_buffer.clear()
        if is_detecting:
            is_detecting = False
            notify_home_assistant(mqtt_client, "off", 0)

if not os.path.exists(MODEL_FILE):
    print(f"Le fichier modèle {MODEL_FILE} n'existe pas.")
    exit()

[classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.load_model(MODEL_FILE)
print(f"Classes du modèle : {classNames}")
print(f"Fenêtre d'analyse : {mtWin} secondes")
print(f"Pas d'analyse : {mtStep} secondes")

try:
    with sd.InputStream(device=DEVICE_INDEX, channels=CHANNELS, samplerate=RATE, blocksize=CHUNK, callback=audio_callback):
        print("Écoute en cours. Appuyez sur Ctrl+C pour arrêter.")
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("Arrêt du programme")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
