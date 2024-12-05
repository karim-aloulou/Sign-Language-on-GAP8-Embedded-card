import cv2
import numpy as np
import pickle

def build_squares(img):
    """
    Dessine une grille de petits rectangles pour capturer la région d'intérêt de la main.
    """
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


def get_hand_hist():
    """
    Capture l'histogramme de la main pour la détection gestuelle et le sauvegarde.
    """
    # Initialisation de la caméra
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erreur : Impossible d'accéder à la caméra.")
        exit()

    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None

    while True:
        ret, img = cam.read()
        if not ret:
            print("Erreur : Impossible de capturer le flux vidéo.")
            break

        # Prétraitement de l'image
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Gestion des interactions clavier
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):		
            # Capture de l'histogramme lorsque 'c' est pressé
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            print("Histogramme capturé.")
        
        elif keypress == ord('q'):
            # Quitter proprement lorsque 'q' est pressé
            print("Sortie du programme.")
            if not flagPressedC:
                print("Attention : Histogramme non capturé. Appuyez sur 'c' pour capturer avant de quitter.")
            break

        if flagPressedC:	
            # Appliquer la rétroprojection si l'histogramme est capturé
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Seuil", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Définir histogramme de la main", img)

    # Libération des ressources
    cam.release()
    cv2.destroyAllWindows()

    # Sauvegarder l'histogramme si capturé
    if flagPressedC:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
            print("Histogramme sauvegardé avec succès.")

# Exécution de la fonction principale
get_hand_hist()
