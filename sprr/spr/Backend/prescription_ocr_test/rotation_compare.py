import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

MODEL_NAME = "microsoft/trocr-base-handwritten"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

rotations = [0, 90, 180, 270]

for filename in sorted(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)
    image = Image.open(image_path).convert("RGB")

    print("\n" + "=" * 60)
    print("IMAGE:", filename)

    for angle in rotations:
        rotated = image.rotate(angle, expand=True)

        pixel_values = processor(
            images=rotated,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print(f"Rotation {angle:3d}° -> {text}")