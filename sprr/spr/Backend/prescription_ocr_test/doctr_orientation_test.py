import os
import numpy as np
from PIL import Image
from doctr.models import page_orientation_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

predictor = page_orientation_predictor(pretrained=True)

for filename in sorted(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    class_idxs, angles, probs = predictor([image_array])

    print("\n" + "=" * 45)
    print("Image:", filename)
    print("Class:", class_idxs)
    print("Angle:", angles)
    print("Probability:", probs)