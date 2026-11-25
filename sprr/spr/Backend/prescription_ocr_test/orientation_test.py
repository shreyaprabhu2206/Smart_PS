
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

for filename in sorted(os.listdir(IMAGE_FOLDER)):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, filename)

    print("\n" + "=" * 50)
    print("Image:", filename)

    command = [
        "tesseract",
        image_path,
        "stdout",
        "--psm", "0",
        "-c", "min_characters_to_try=5"
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        print(result.stdout.strip())

        if result.stderr.strip():
            print("Warnings:", result.stderr.strip())

    except Exception as error:
        print("Error:", error)
