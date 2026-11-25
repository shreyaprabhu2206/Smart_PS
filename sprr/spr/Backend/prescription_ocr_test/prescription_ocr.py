
import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import wer

# Configuration
MODEL_NAME = "microsoft/trocr-base-handwritten"
IMAGE_FOLDER = "cropped"
RESULT_FILE = "cropped_results.txt"
GROUND_TRUTH_FILE = "ground_truth.txt"

BEAM_SETTINGS = [1, 4, 6]

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
print("Loading TrOCR Base...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Model loaded successfully!\n")

# Read verified text
ground_truth = {}

with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if "|" in line:
            filename, text = line.split("|", 1)
            ground_truth[filename] = text

# Process images
predictions = {}
results = []

for filename in sorted(os.listdir(IMAGE_FOLDER)):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    try:
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

        print("Image:", filename)

        beam_predictions = {}

        # Test different beam settings
        for beams in BEAM_SETTINGS:

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=128,
                    num_beams=beams
                )

            predicted_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            beam_predictions[beams] = predicted_text

            print(f"Beams {beams}: {predicted_text}")

        # Use 4 beams as the baseline for WER
        predictions[filename] = beam_predictions[4]

        if filename in ground_truth:
            print("Actual:", ground_truth[filename])

        print("-" * 60)

    except Exception as error:
        print("Error processing", filename, ":", error)

# Calculate WER using 4-beam predictions
for filename, actual_text in ground_truth.items():

    if filename in predictions:
        results.append((actual_text, predictions[filename]))

if results:
    actual_texts = [item[0] for item in results]
    predicted_texts = [item[1] for item in results]

    # Normalize capitalization and whitespace
    actual_normalized = [text.lower().strip() for text in actual_texts]
    predicted_normalized = [text.lower().strip() for text in predicted_texts]

    error_rate = wer(actual_normalized, predicted_normalized)

    print("\n========== RESULTS (4 BEAMS) ==========")
    print("Images evaluated:", len(results))
    print("Word Error Rate:", round(error_rate * 100, 2), "%")

else:
    print("No matching images and ground-truth entries found.")

# Save predictions for all beam settings
with open(RESULT_FILE, "w", encoding="utf-8") as file:

    for filename in sorted(predictions):

        file.write(f"{filename}\n")

        # Save the baseline prediction
        file.write(f"4-beam prediction|{predictions[filename]}\n")

print("\nPredictions saved to", RESULT_FILE)
