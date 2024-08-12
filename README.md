# sound-detection
Détection de sonnette avec Raspberry Pi, Respeaker 4-array Seed et Home AssistantMatériel nécessaire :

Raspberry Pi 3B+
Microphone Respeaker 4-array Seed
Carte SD (min. 8 Go)
Connexion Internet
Étape 1 : Configuration initiale de la Raspberry Pi

Téléchargez et installez Raspberry Pi OS sur votre carte SD.
Insérez la carte SD dans la Raspberry Pi et démarrez-la.
Connectez-vous à la Raspberry Pi (par défaut, utilisateur : pi, mot de passe : raspberry).
Exécutez sudo raspi-config pour configurer le Wi-Fi, changer le mot de passe, etc.
Étape 2 : Installation et configuration du Respeaker 4-array Seed

Connectez le Respeaker 4-array Seed à votre Raspberry Pi.
Ouvrez un terminal et exécutez :
sudo apt-get update
sudo apt-get install git
git clone https://github.com/respeaker/seeed-voicecard.git
cd seeed-voicecard
sudo ./install.sh
sudo reboot
Après le redémarrage, vérifiez que le microphone est détecté :
arecord -l
Étape 3 : Installation des dépendances

sudo apt-get install python3-pip portaudio19-dev
pip3 install pyaudio numpy paho-mqtt scipy sounddevice
pip3 install -U scikit-learn
pip3 install pyAudioAnalysis
Étape 4 : Collecte des échantillons audio

Créez deux répertoires :
mkdir sonnette
mkdir non-sonnette
Enregistrez 10-15 échantillons de sonnette dans le répertoire « sonnette » :
arecord -d 2 -f cd -t wav sonnette/sonnette_01.wav
Répétez cette commande pour chaque échantillon, en changeant le nom du fichier.

Enregistrez 10-15 échantillons de bruits environnants dans le répertoire « non-sonnette » de la même manière.
Étape 5 : Entraînement du modèle

Créez un script Python nommé train_model.py :

from pyAudioAnalysis import audioTrainTest as aT

aT.extract_features_and_train(["sonnette", "non-sonnette"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "doorbell_model", False)
Exécutez le script :

python3 train_model.py
Étape 6 : Création du script de détection

Créez un fichier doorbell_detection.py :

import pyaudio
import numpy as np
import time
import paho.mqtt.client as mqtt
from pyAudioAnalysis import audioTrainTest as aT
import scipy.fft
import tempfile
import sounddevice as sd
import wave

# Paramètres audio
CHUNK = 11025
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Paramètres de détection
MIN_PROB_THRESHOLD = 0.6
MIN_POWER_THRESHOLD = 0.02
DETECTION_WINDOW = 3

# Paramètres MQTT
MQTT_BROKER = "adresse_ip_de_home_assistant"
MQTT_PORT = 1883
MQTT_TOPIC = "doorbell/state"
MQTT_USER = "votre_utilisateur_mqtt"
MQTT_PASSWORD = "votre_mot_de_passe_mqtt"

# Variables globales
detection_buffer = []
is_detecting = False
detection_start_time = 0

def on_connect(client, userdata, flags, rc):
    print(f"Connecté au broker MQTT avec le code : {rc}")

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.on_connect = on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

def record_audio(duration, fs, channels):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    return recording

def save_audio(audio, filename, fs):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

def analyze_frequency(audio):
    fft = scipy.fft.fft(audio)
    freqs = scipy.fft.fftfreq(len(audio), 1/RATE)
    peak_freq = freqs[np.argmax(np.abs(fft))]
    return abs(peak_freq)

def classify_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        save_audio(audio, temp_file.name, RATE)
        features = aT.extract_features(temp_file.name, RATE, CHUNK, CHANNELS)
        result = aT.file_classification(features, "doorbell_model", "svm")
    return result[1][0]  # Probabilité de la classe "sonnette"

def notify_home_assistant(client, state, probability):
    message = "on" if state == "on" else "off"
    client.publish(MQTT_TOPIC, message)
    print(f"Notification envoyée à Home Assistant : Sonnette {message} (Probabilité: {probability:.2f})")

def audio_callback(indata, frame_count, time_info, status):
    global detection_buffer, is_detecting, detection_start_time

    audio = np.frombuffer(indata, dtype=np.float32)
    audio_power = np.mean(np.abs(audio))
    print(f"Puissance audio: {audio_power:.4f}")

    if audio_power >= MIN_POWER_THRESHOLD:
        probability = classify_audio(audio)
        peak_freq = analyze_frequency(audio)
        print(f"Probabilité: {probability:.2f}, Fréquence de pic: {peak_freq:.2f} Hz")

        detection_buffer.append((probability, audio_power, peak_freq))
        if len(detection_buffer) > DETECTION_WINDOW:
            detection_buffer.pop(0)

        if any(p >= MIN_PROB_THRESHOLD for p, _, _ in detection_buffer):
            if not is_detecting:
                is_detecting = True
                detection_start_time = time.time()
                notify_home_assistant(client, "on", max(p for p, _, _ in detection_buffer))
        elif is_detecting:
            detection_duration = time.time() - detection_start_time
            if detection_duration >= 0.5:  # Durée minimale de détection
                is_detecting = False
                notify_home_assistant(client, "off", max(p for p, _, _ in detection_buffer))

    else:
        print("Puissance audio insuffisante")
        detection_buffer.clear()
        if is_detecting:
            is_detecting = False
            notify_home_assistant(client, "off", 0)

    return (indata, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

print("Démarrage de la détection de sonnette...")
stream.start_stream()

try:
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Arrêt de la détection")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    client.disconnect()
Étape 7 : Configuration du service systemdCréez un fichier /etc/systemd/system/doorbell-detection.service :

[Unit]
Description=Doorbell Detection Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/doorbell_detection.py
WorkingDirectory=/home/pi
User=pi
Group=pi
Restart=always
RestartSec=10s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
Activez et démarrez le service :

sudo systemctl daemon-reload
sudo systemctl enable doorbell-detection.service
sudo systemctl start doorbell-detection.service
Étape 8 : Configuration de Home Assistant

Assurez-vous que l’intégration MQTT est configurée dans Home Assistant.
Ajoutez un capteur binaire dans votre fichier configuration.yaml :
binary_sensor:
  - platform: mqtt
    name: "Sonnette"
    state_topic: "doorbell/state"
    payload_on: "on"
    payload_off: "off"
    device_class: sound
Redémarrez Home Assistant pour appliquer les changements.
Ce tutoriel complet couvre toutes les étapes nécessaires, de la configuration initiale à l’entraînement du modèle avec pyAudioAnalysis et à la mise en place du système de détection. N’oubliez pas d’ajuster les paramètres (comme MIN_PROB_THRESHOLD) en fonction de vos besoins spécifiques et des résultats obtenus lors des tests.
