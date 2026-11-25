import os
import torch
from PIL import Image, ImageOps, ImageEnhance
from jiwer import wer, cer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
PREPROCESSED_FOLDER = os.path.join(BASE_DIR, "preprocessed")

MODEL_NAME = "microsoft/trocr-base-handwritten"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

ground_truth = {}

with open(os.path.join(BASE_DIR, "ground_truth.txt"), encoding="utf-8") as file:
    for line in file:
        filename, text = line.strip().split("|", 1)
        ground_truth[filename] = text


def recognize(image):
    pixel_values = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4
        )

    return processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]


def preprocess(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.3)

    width, height = image.size
    image = image.resize(
        (width * 2, height * 2),
        Image.Resampling.LANCZOS
    )

    return image.convert("RGB")


strategies = {
    "Original": [],
    "Preprocessed": [],
    "Manual_Rotation": [],
}

references = []

for filename in sorted(ground_truth):
    image_path = os.path.join(IMAGE_FOLDER, filename)

    original = Image.open(image_path).convert("RGB")
    processed = preprocess(original)

    # Original
    pred_original = recognize(original)

    # Preprocessed
    pred_preprocessed = recognize(processed)

    # Manually selected rotation from previous experiment
    manual_angles = {
        "prescription_01.png": 90,
        "prescription_02.png": 90,
        "prescription_03.png": 90,
        "prescription_04.png": 90,
        "prescription_05.png": 0,
        "prescription_06.png": 0,
        "prescription_07.png": 0,
        "prescription_08.png": 90,
        "prescription_09.png": 90,
    }

    rotated = original.rotate(manual_angles[filename], expand=True)
    pred_rotation = recognize(rotated)

    reference = ground_truth[filename]

    references.append(reference)
    strategies["Original"].append(pred_original)
    strategies["Preprocessed"].append(pred_preprocessed)
    strategies["Manual_Rotation"].append(pred_rotation)

    print("\n" + "=" * 50)
    print("Image:", filename)
    print("Ground truth:", reference)
    print("Original:", pred_original)
    print("Preprocessed:", pred_preprocessed)
    print("Manual rotation:", pred_rotation)


print("\n\nFINAL COMPARISON")
print("=" * 50)

for strategy, predictions in strategies.items():
    print(f"\n{strategy}")
    print("WER:", wer(references, predictions))
    print("CER:", cer(references, predictions))