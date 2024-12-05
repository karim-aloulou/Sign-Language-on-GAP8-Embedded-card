import tensorflow as tf

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\msi\\Desktop\\isep\\CV\\Sign-Language-Interpreter-using-Deep-Learning\\Code\\cnn_model.tflite")
interpreter.allocate_tensors()

# Obtenir les informations sur les entrées et sorties
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Entrées :", input_details)
print("Sorties :", output_details)

# Charger un exemple d'image
import numpy as np
test_image = np.random.rand(1, 50, 50, 1).astype(np.float32)  # Générer une image aléatoire

# Faire une prédiction
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Prédiction :", output_data)
