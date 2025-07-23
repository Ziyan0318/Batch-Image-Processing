import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from mtcnn import MTCNN

def crop_face(image, cut_percentage):
    detector = MTCNN()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if not faces:
        print("No face detected")
        return image

    # Find main face
    def area(face):
        x, y, w, h = face['box']
        return w * h
    main_face = max(faces, key=area)
    x, y, w, h = main_face['box']
    x, y = max(0, x), max(0, y)

    # Crop bottom portion of face
    crop_top = int(y + (cut_percentage / 100.0) * h)
    cropped_img = image[crop_top:, :]
    return cropped_img

def remove_background(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    no_bg = remove(pil_image)
    return no_bg

def resize_and_crop_center(image_pil, target_size):
    target_w, target_h = target_size
    img_w, img_h = image_pil.size

    scale = max(target_w / img_w, target_h / img_h)
    resized = image_pil.resize((int(img_w * scale), int(img_h * scale)))

    # Center crop
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    cropped = resized.crop((left, top, right, bottom))
    return cropped

def process_image(image_path, actions, cut_percent, resize_dims):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read: {image_path}")
        return None, os.path.basename(image_path)

    if '1' in actions:
        image = crop_face(image, cut_percent)

    if '2' in actions:
        image = remove_background(image)

    if '3' in actions:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = resize_and_crop_center(image, resize_dims)

    return image, os.path.basename(image_path)

def main():
    mode = input("Process a single image or a folder? (Enter 'image' or 'folder'): ").strip().lower()

    if mode == 'image':
        paths = [input("Enter the image file path: ").strip()]
    elif mode == 'folder':
        folder_path = input("Enter the folder path: ").strip()
        paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    else:
        print("Invalid input. Please enter 'image' or 'folder'.")
        return

    actions = input("Choose actions - 1: crop face, 2: remove background, 3: resize (e.g. 1 3): ").split()
    cut_percent = 60
    if '1' in actions:
        cut_percent = float(input("Enter percentage of face to crop (e.g. 60): "))

    resize_dims = (1350, 1800)
    if '3' in actions:
        width = int(input("Enter final width (e.g. 1350): "))
        height = int(input("Enter final height (e.g. 1800): "))
        resize_dims = (width, height)

    output_dir = input("Enter the output folder: ").strip()
    os.makedirs(output_dir, exist_ok=True)

    for path in paths:
        result, name = process_image(path, actions, cut_percent, resize_dims)
        if result is None:
            continue

        save_name = os.path.splitext(name)[0] + '_final.png'
        save_path = os.path.join(output_dir, save_name)

        if isinstance(result, Image.Image):
            result.save(save_path)
        else:
            cv2.imwrite(save_path, result)

        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    main()
