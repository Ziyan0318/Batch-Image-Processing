#Import tools and models
from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
import os

image_path = 'DoubleTakeDoubleTakeFullSizeNotchedCapSleeveKnitTop/DoubleTakeDoubleTakeFullSizeNotchedCapSleeveKnitTop0.jpg' #upload image
image = cv2.imread(image_path) # use opencv to read image (in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # MTCNN required RGB, convert BGR to RGB format

detector = MTCNN() # Initialize MTCNN detector

# detect faces in the image
faces = detector.detect_faces(image_rgb)
if not faces:
    print("No face detected")
else:
    def area(face):
        x, y, w, h = face['box']
        return w * h
    main_face = max(faces, key=area)
    x, y, w, h = main_face['box']
    x, y = max(0, x), max(0, y) # ensure there's no negative coordinate

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw frame

    # save result
    output_path = 'output_mainface.jpg'
    cv2.imwrite(output_path, image)
    print(f"Detected done, face frame saved: {output_path}")