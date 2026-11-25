
import os
import torch
from PIL import Image, ImageOps, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import wer, cer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth.txt")

MODEL_NAME = "microsoft/trocr-base-handwritten"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading TrOCR Base...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

print("Model loaded!\n")

# Read ground truth
ground_truth = {}

with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if "|" in line:
            filename, text = line.split("|", 1)
            ground_truth[filename] = text

# Four preprocessing methods
def preprocess(image, method):

    image = image.convert("RGB")

    if method == "A":
        # Grayscale only
        image = ImageOps.grayscale(image)

    elif method == "B":
        # Grayscale + autocontrast
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image, cutoff=1)

    elif method == "C":
        # Grayscale + contrast enhancement
        image = ImageOps.grayscale(image)
        image = ImageEnhance.Contrast(image).enhance(1.3)

    elif method == "D":
        # Full preprocessing
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image, cutoff=1)
        image = ImageEnhance.Contrast(image).enhance(1.3)

        width, height = image.size
        image = image.resize(
            (width * 2, height * 2),
            Image.Resampling.LANCZOS
        )

    return image.convert("RGB")


methods = {
    "A": "Grayscale only",
    "B": "Grayscale + Autocontrast",
    "C": "Grayscale + Contrast",
    "D": "Full preprocessing"
}

predictions = {method: {} for method in methods}

# Process images
for filename in sorted(ground_truth):

    image_path = os.path.join(IMAGE_FOLDER, filename)

    if not os.path.exists(image_path):
        print("Missing image:", filename)
        continue

    original = Image.open(image_path).convert("RGB")

    print("\nImage:", filename)

    for method in methods:

        processed = preprocess(original, method)

        pixel_values = processor(
            images=processed,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        predictions[method][filename] = prediction

        print(f"{method} ({methods[method]}): {prediction}")

    print("Actual:", ground_truth[filename])

# Evaluate methods
print("\n========== COMPARISON RESULTS ==========")

for method, description in methods.items():

    actual = []
    predicted = []

    for filename, correct_text in ground_truth.items():

        if filename in predictions[method]:

            actual.append(correct_text.lower().strip())
            predicted.append(
                predictions[method][filename].lower().strip()
            )

    if actual:
        word_error = wer(actual, predicted)
        char_error = cer(actual, predicted)

        print(f"\n{method}: {description}")
        print(f"WER: {word_error * 100:.2f}%")
        print(f"CER: {char_error * 100:.2f}%")

# Save results
output_file = os.path.join(
    BASE_DIR,
    "preprocessing_comparison_results.txt"
)

with open(output_file, "w", encoding="utf-8") as file:

    for method, description in methods.items():

        file.write(f"\n{method}: {description}\n")

        for filename, prediction in predictions[method].items():
            file.write(f"{filename}|{prediction}\n")

print("\nResults saved to:", output_file)
