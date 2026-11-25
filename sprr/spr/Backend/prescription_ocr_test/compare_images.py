
import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import wer

MODEL_NAME = "microsoft/trocr-base-handwritten"

FOLDERS = ["images", "preprocessed", "cropped"]
GROUND_TRUTH_FILE = "ground_truth.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
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

# Store predictions
all_predictions = {}

for folder in FOLDERS:

    print(f"\n========== {folder.upper()} ==========")

    all_predictions[folder] = {}

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for filename in sorted(os.listdir(folder)):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")

            pixel_values = processor(
                images=image,
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

            all_predictions[folder][filename] = prediction

            print("\nImage:", filename)
            print("Predicted:", prediction)

            if filename in ground_truth:
                print("Actual:", ground_truth[filename])

        except Exception as error:
            print("Error:", filename, error)

# Calculate WER separately for each folder
print("\n========== WER COMPARISON ==========")

for folder in FOLDERS:

    actual_texts = []
    predicted_texts = []

    for filename, actual in ground_truth.items():

        if filename in all_predictions[folder]:

            actual_texts.append(actual.lower().strip())
            predicted_texts.append(
                all_predictions[folder][filename].lower().strip()
            )

    if actual_texts:
        error_rate = wer(actual_texts, predicted_texts)

        print(
            f"{folder}: {error_rate * 100:.2f}% WER"
        )
    else:
        print(f"{folder}: No matching images")

# Save comparison
with open("image_comparison.txt", "w", encoding="utf-8") as file:

    for folder in FOLDERS:

        file.write(f"\n========== {folder.upper()} ==========\n")

        for filename, prediction in all_predictions[folder].items():
            file.write(f"{filename}|{prediction}\n")

print("\nComparison saved to image_comparison.txt")
