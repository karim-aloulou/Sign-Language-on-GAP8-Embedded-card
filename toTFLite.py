import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Charger le modèle
model_path = r"C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\cnn_model_keras2.keras"

# Vérifier si le fichier existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier du modèle n'existe pas : {model_path}")

model = load_model(model_path)

# Convertir le modèle en TFLite avec quantification
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Pas d'optimisation
converter.optimizations = []
try:
    tflite_model = converter.convert()
    print("Conversion réussie en TensorFlow Lite.")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la conversion en TFLite : {e}")

# Sauvegarder le modèle TFLite
output_path = "cnn_model.tflite"
try:
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Le modèle TFLite a été sauvegardé à : {os.path.abspath(output_path)}")
except Exception as e:
    raise RuntimeError(f"Erreur lors de l'enregistrement du modèle TFLite : {e}")
