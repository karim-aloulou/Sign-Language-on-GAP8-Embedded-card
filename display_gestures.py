import cv2
import os
import random
import numpy as np

def get_image_size():
    return 200, 200  # Taille augmentée

# Liste des gestes
gestures_path = r'C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\gestures'
gestures = [g for g in os.listdir(gestures_path) if os.path.isdir(os.path.join(gestures_path, g))]
gestures.sort(key=int)

begin_index = 0
end_index = min(3, len(gestures))  # Ajuste le nombre d'images par ligne
image_x, image_y = get_image_size()

# Nombre de lignes pour l'affichage
rows = (len(gestures) + 2) // 3

full_img = None
for i in range(rows):
    col_img = None
    for j in range(begin_index, end_index):
        # Liste des fichiers existants dans le sous-dossier
        img_dir = f"{gestures_path}/{j}"
        if not os.path.exists(img_dir):
            print(f"Sous-dossier introuvable : {img_dir}")
            continue

        img_files = os.listdir(img_dir)
        if not img_files:
            print(f"Aucune image dans le sous-dossier : {img_dir}")
            continue

        # Charger une image aléatoire existante
        img_name = random.choice(img_files)
        img_path = os.path.join(img_dir, img_name)
        print(f"Traitement de l'image : {img_path}")

        img = cv2.imread(img_path, 0)
        if img is None:
            print(f"Erreur de lecture : {img_path}. Image vide créée.")
            img = np.zeros((image_y, image_x), dtype=np.uint8)
        else:
            img = cv2.resize(img, (image_x, image_y))

        if col_img is None:
            col_img = img
        else:
            col_img = np.hstack((col_img, img))

    begin_index += 3
    end_index += 3
    if full_img is None:
        full_img = col_img
    else:
        full_img = np.vstack((full_img, col_img))

# Agrandir dynamiquement la grille finale
scale_factor = 4  # Ajustez selon vos besoins
full_img = cv2.resize(full_img, (full_img.shape[1] * scale_factor, full_img.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)

cv2.imshow("gestures", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)
