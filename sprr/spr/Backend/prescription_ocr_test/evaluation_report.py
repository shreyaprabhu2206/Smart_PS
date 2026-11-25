import os
import pandas as pd
from jiwer import wer, cer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ground_truth = {}
with open(os.path.join(BASE_DIR, "ground_truth.txt"), encoding="utf-8") as file:
    for line in file:
        filename, text = line.strip().split("|", 1)
        ground_truth[filename] = text

predictions = {
    "prescription_01.png": "Ashwini Clinic",
    "prescription_02.png": "Ashwini Clinic",
    "prescription_03.png": "America Institute of Technology",
    "prescription_04.png": "atria .",
    "prescription_05.png": "3 Books contain papers .",
    "prescription_06.png": "# Satu is the greatest",
    "prescription_07.png": "Bottle has water in it .",
    "prescription_08.png": "books contain Paper .",
    "prescription_09.png": "Books contain father .",
}

rows = []

for filename, reference in ground_truth.items():
    prediction = predictions[filename]

    rows.append({
        "Image": filename,
        "Ground Truth": reference,
        "Prediction": prediction,
        "WER": wer(reference, prediction),
        "CER": cer(reference, prediction),
    })

df = pd.DataFrame(rows)

print(df.to_string(index=False))

output_path = os.path.join(BASE_DIR, "evaluation_report.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print("\nReport saved to:", output_path)