#Import tools and models
from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
import os

image_path = input("Enter image path: ") #upload image
image = cv2.imread(image_path) # use opencv to read image (in BGR format)

# check if image could be found
if image is None:
    print(f"Failed to read image, please check if the path is correct: {image_path}")
    exit() # program end

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # MTCNN required RGB, convert BGR to RGB format

detector = MTCNN() # Initialize MTCNN detector

# detect faces in the image
faces = detector.detect_faces(image_rgb)
if not faces:
    print("No face detected")
else:
    # find model's face
    def area(face):
        x, y, w, h = face['box']
        return w * h
    main_face = max(faces, key=area)

    # x, y are the coordinates of the top left corner of face box, w = wideth, h = height
    x, y, w, h = main_face['box']
    x, y = max(0, x), max(0, y) # ensure there's no negative coordinate

    """
    # draw frame
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # save result
    output_path = 'output_mainface.jpg'
    cv2.imwrite(output_path, image)
    print(f"Detected done, face frame saved: {output_path}")
    """

    # Crop 60% of model's face
    crop_top = int(y + 0.6 * h)
    crop_img = image[crop_top:, :]
    cropped_output_path = 'output_cropped.jpg'
    # save result
    cv2.imwrite(cropped_output_path, crop_img)
    print(f"Cropped image saved: {cropped_output_path}")