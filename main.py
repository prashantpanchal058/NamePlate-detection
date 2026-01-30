from fastapi import FastAPI, File, UploadFile
import uvicorn
from ultralytics import YOLO
import cv2
import easyocr
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("..\models\\best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

def extract_coordinates(image_path):
    results = model.predict(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            return int(x1), int(y1), int(x2), int(y2)

def extract_text_from_plate(image_path):
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = extract_coordinates(image_path)
    cropped = img[y1:y2, x1:x2]
    result = reader.readtext(cropped)
    for bbox, text, confidence in result:
        return text, float(confidence)
    return None, 0.0  # if nothing detected

# Convert uploaded file to a saved temp image file
def save_temp_image(data: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(data)
    temp_file.close()
    return temp_file.name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and save uploaded image
    contents = await file.read()
    image_path = save_temp_image(contents)

    # Detect and extract text from license plate
    try:
        text, confidence = extract_text_from_plate(image_path)
        print(confidence)
        confidence = round(confidence, 2) * 100
        return {
            "detected_text": text,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
