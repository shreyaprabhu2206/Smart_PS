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


def get_prediction_and_score(image):
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            output_scores=True,
            return_dict_in_generate=True
        )

    text = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0]

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True
    )

    score = transition_scores.sum().item()
    token_count = transition_scores.shape[1]

    average_score = score / max(token_count, 1)

    return text, average_score


for filename in sorted(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    original = Image.open(image_path).convert("RGB")

    print("\n" + "=" * 50)
    print("IMAGE:", filename)

    results = []

    for angle in [0, 90, 180, 270]:
        rotated = original.rotate(angle, expand=True)

        text, score = get_prediction_and_score(rotated)

        results.append((angle, text, score))

        print(f"Rotation {angle:3}° -> {text}")
        print(f"Average score: {score:.4f}")

    best = max(results, key=lambda item: item[2])

    print("\nSelected rotation:", best[0])
    print("Selected text:", best[1])