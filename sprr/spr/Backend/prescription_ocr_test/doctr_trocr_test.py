import os
import numpy as np
import torch
from PIL import Image
from doctr.models import page_orientation_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

device = "cuda" if torch.cuda.is_available() else "cpu"

orientation_model = page_orientation_predictor(pretrained=True)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(device)

model.eval()

for filename in sorted(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = Image.open(image_path).convert("RGB")

    class_idxs, angles, probs = orientation_model(
        [np.array(image)]
    )

    angle = int(angles[0])

    # Apply the predicted correction
    corrected = image.rotate(angle, expand=True)

    pixel_values = processor(
        images=corrected,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    text = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    print("\n" + "=" * 45)
    print("Image:", filename)
    print("Predicted angle:", angle)
    print("Probability:", float(probs[0]))
    print("Recognized text:", text)