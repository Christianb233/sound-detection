from pyAudioAnalysis import audioTrainTest as aT

# Entraînement du modèle
aT.extract_features_and_train(["sonnette", "non_sonnette"], 1.0, 1.0,
                              aT.shortTermWindow, aT.shortTermStep,
                              "svm", "/home/pi/pyAudioAnalysis/sonnette_model", False)

