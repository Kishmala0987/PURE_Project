import os
import zipfile
import gdown

FILE_ID = "1IMnmCIQ283JNeavNvOkfVYslMHVZx29Z"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "data")
ZIP_PATH = os.path.join(OUTPUT_DIR, "data.zip")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading dataset from Google Drive...")

gdown.download(
    id=FILE_ID,
    output=ZIP_PATH,
    quiet=False
)

print("Extracting dataset...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)

os.remove(ZIP_PATH)

print("Dataset downloaded and extracted successfully!")
