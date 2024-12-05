import cv2
import os

def flip_images():
    # Use raw string or double backslashes to fix path issue
    gest_folder = r"C:\Users\msi\Desktop\isep\CV\Sign-Language-Interpreter-using-Deep-Learning\Code\gestures"
    # gest_folder = "C:\\Users\\msi\\Desktop\\isep\\CV\\Sign-Language-Interpreter-using-Deep-Learning\\Code\\gestures"
    
    if not os.path.exists(gest_folder):
        print(f"Error: Folder '{gest_folder}' does not exist.")
        return
    
    for g_id in os.listdir(gest_folder):
        gesture_path = os.path.join(gest_folder, g_id)
        if not os.path.isdir(gesture_path):
            print(f"Skipping non-directory: {gesture_path}")
            continue
        
        for i in range(1200):
            path = os.path.join(gesture_path, f"{i+1}.jpg")
            new_path = os.path.join(gesture_path, f"{i+1+1200}.jpg")
            print(f"Processing: {path}")
            
            # Check if the file exists
            if not os.path.exists(path):
                print(f"File not found: {path}. Skipping.")
                continue
            
            # Read the image
            img = cv2.imread(path, 0)
            if img is None:
                print(f"Error reading image: {path}. Skipping.")
                continue
            
            # Flip the image
            flipped_img = cv2.flip(img, 1)
            
            # Save the flipped image
            cv2.imwrite(new_path, flipped_img)
            print(f"Saved flipped image to: {new_path}")

flip_images()
