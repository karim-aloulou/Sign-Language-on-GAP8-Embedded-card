import cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
import pyttsx3
from keras.models import load_model
from threading import Thread, Lock

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
tts_lock = Lock()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model_path = r'C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\cnn_model_keras2.keras'
if not os.path.exists(model_path):
    print("Error: Model file not found. Exiting...")
    exit()

model = load_model(model_path)

# Load the hand histogram
def get_hand_hist():
    hist_path = "hist"
    if not os.path.exists(hist_path):
        print("Error: Hand histogram file not found. Exiting...")
        exit()
    with open(hist_path, "rb") as f:
        return pickle.load(f)

# Get image size for preprocessing
def get_image_size():
    sample_image = r'C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\gestures\20\100.jpg'
    if not os.path.exists(sample_image):
        print("Error: Sample image not found. Exiting...")
        exit()
    img = cv2.imread(sample_image, 0)
    return img.shape

image_x, image_y = get_image_size()
hist = get_hand_hist()

# Preprocess image for the model
def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

# Predict using the loaded model
def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

# Fetch gesture name from the database
def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    try:
        cmd = "SELECT g_name FROM gesture WHERE g_id = ?"
        cursor = conn.execute(cmd, (pred_class,))
        result = cursor.fetchone()
        return result[0] if result else "Unknown"
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Error"
    finally:
        conn.close()

# Text-to-speech function
def say_text(text):
    if not is_voice_on:
        return
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

# Get image contours and threshold
def get_img_contour_thresh(img, x, y, w, h):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.medianBlur(cv2.GaussianBlur(dst, (11, 11), 0), 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.cvtColor(cv2.merge((thresh, thresh, thresh)), cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return img, contours, thresh

# Text mode for gesture recognition
def text_mode(cam):
    global is_voice_on
    x, y, w, h = 300, 100, 300, 300
    text, word, count_same_frame = "", "", 0
    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Camera frame not accessible.")
            break

        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img, x, y, w, h)
        old_text = text
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_probab, pred_class = keras_predict(model, thresh)
                text = get_pred_text_from_db(pred_class)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0
                if count_same_frame > 20 and len(text) == 1:
                    Thread(target=say_text, args=(text,)).start()
                    word += text
                    count_same_frame = 0
        else:
            text = ""

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
        cv2.putText(blackboard, f"Predicted text: {text}", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.imshow("Recognizing gesture", np.hstack((img, blackboard)))

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):  # Quit with 'q'
            break
        elif keypress == ord('v'):
            is_voice_on = not is_voice_on

def recognize():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Camera not accessible.")
        return
    text_mode(cam)
    cam.release()
    cv2.destroyAllWindows()

# Initialize variables
is_voice_on = True
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))  # Warm up model
recognize()
