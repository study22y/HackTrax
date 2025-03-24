from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
import pytesseract
from PIL import Image
from transformers import pipeline
import os
from werkzeug.utils import secure_filename
import fitz 
import spacy
import cv2
import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import pipeline
from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from typing import List
import firebase_admin
from flask_cors import CORS
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'outputs'
PROCESSED_FOLDER = 'static/processed'  # Corrected this line
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

# Flask config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER  # Corrected this line

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # N



try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading en_core_web_lg model...")
    os.system("spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

transformers_model = pipeline(
    "token-classification",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="average",
    ignore_labels=["O", "MISC"]
)

# ✅ Custom Entity Recognizer
class TransformersRecognizer(EntityRecognizer):
    def __init__(self, model_pipeline, supported_language="en"):
        self.pipeline = model_pipeline
        self.label2presidio = {
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "MISC": "MISC",
        }
        super().__init__(supported_entities=list(self.label2presidio.values()), supported_language=supported_language)

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None) -> List[RecognizerResult]:
        results = []
        predicted_entities = self.pipeline(text)

        for e in predicted_entities:
            converted_entity = self.label2presidio.get(e["entity_group"], None)
            if converted_entity and (entities is None or converted_entity in entities):
                results.append(RecognizerResult(entity_type=converted_entity, start=e["start"], end=e["end"], score=e["score"]))
        return results

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

transformers_recognizer = TransformersRecognizer(transformers_model)
analyzer.registry.add_recognizer(transformers_recognizer)

COLOR_MASK = "BLACK"
color_map = {
    "BLACK": (0, 0, 0),
    "WHITE": (1.0, 1.0, 1.0),
    "RED": (1.0, 0.0, 0.0),
    "GREEN": (0.0, 1.0, 0.0),
    "BLUE": (0.0, 0.0, 1.0),
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text, doc

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


def allowed_file(filename):
    """Check if file type is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

import re
import uuid

def generate_secure_token():
    """Generate a unique secure token"""
    return str(uuid.uuid4())


def indian_specific_regex(text):
    """Returns additional sensitive data based on Indian-specific regex patterns"""
    
    regex_patterns = {
        "IN_PAN": r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",  
        "IN_AADHAAR": r"\b\d{4} \d{4} \d{4}\b", 
        "IN_PHONE": r"\b(?:\+91|91)?\d{10}\b",  
        "IN_PHONE_WITHOUT_CODE":r"\b[6789]\d{9}\b",
        # "IN_BANK_ACCOUNT": r"\b\d{9,18}\b"
    }
    
    sensitive_data = {}
    
    for entity_name, pattern in regex_patterns.items():
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entity_text = match.group(0)
            0
            safe_token = generate_secure_token()
            sensitive_data[entity_text] = {
                "entity": entity_name,
                "safe_token": safe_token,
                "confidence_score": 1 
            }
    
    return sensitive_data
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def get_sensitive_data(text):
    analysis_results = analyzer.analyze(text=text, entities=None, language="en")

    sensitive_data = {}
    

    for result in analysis_results:
        entity_text = text[result.start:result.end]  # Extract actual sensitive text
        if len(entity_text) <= 2:
            continue
        entity_label = result.entity_type  # Get entity label

        # Generate a secure token for the entity
        safe_token = generate_secure_token()

        # Get confidence score
        confidence_score = result.score
        if entity_label=="IN_PAN" and confidence_score<=0.7:
            continue
        # Store in dictionary
        else:
            sensitive_data[entity_text] = {
            "entity": entity_label,
            "safe_token": safe_token,
            "confidence_score": confidence_score
        }
    regex_sensitive_data = indian_specific_regex(text)

    # Merge the results
    sensitive_data.update(regex_sensitive_data)

    return sensitive_data
@app.route('/extract-sensitive-data', methods=['POST'])
def extract_and_store_sensitive_data(text):
    try:
        # Parse request data
       
        # Extract sensitive data
        sensitive_data = get_sensitive_data(text)

        # Store in Firestore
        store_sensitive_data_firestore(sensitive_data)

        return jsonify({"message": "Data saved successfully!", "sensitive_data": sensitive_data})

    except Exception as e:
        print("Error occurred:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500


def convert_np_floats(value):
    """Recursively converts numpy float32 values to Python float."""
    if isinstance(value, np.float32):
        return float(value)
    if isinstance(value, dict):
        return {k: convert_np_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_np_floats(v) for v in value]
    return value

def store_sensitive_data_firestore(sensitive_data):
    try:
        print(f"Received sensitive data: {sensitive_data}")
        if not sensitive_data:
            print("No sensitive data to store.")
            return

        for entity_text, details in sensitive_data.items():
            safe_token = details.get("safe_token")

            if not safe_token:
                print(f"Skipping {entity_text} due to missing safe_token")
                continue

            cleaned_details = convert_np_floats(details)

            print(f"Storing entity: {entity_text}, Safe Token: {safe_token}")

            try:
                db.collection("tokens").document(safe_token).set({
                    "original_text": entity_text,
                    "entity": cleaned_details.get("entity"),
                    "confidence_score": cleaned_details.get("confidence_score")
                })
                print(f"✅ Stored {entity_text} successfully!")
            except Exception as firestore_error:
                print(f"❌ Firestore Error for {entity_text}: {firestore_error}")

    except Exception as e:
        print(f"Error storing data in Firestore: {e}")


def apply_blur(page, area):
    """Apply a blur effect on a specific area."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    x0, y0, x1, y1 = map(int, [area.x0, area.y0, area.x1, area.y1])
    sub_img = img[y0:y1, x0:x1]
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(sub_img, (21, 21), 30)
    img[y0:y1, x0:x1] = blurred

    # Convert back to PDF
    pix.samples = img.tobytes()
    page.insert_image(area, pixmap=pix)
    
def apply_blur_on_image(image_path, area):
    """Apply a blur effect on a specific area in the image."""
    img = cv2.imread(image_path)
    
    x0, y0, x1, y1 = map(int, [area[0], area[1], area[2], area[3]])
    sub_img = img[y0:y1, x0:x1]
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(sub_img, (21, 21), 30)
    img[y0:y1, x0:x1] = blurred

    cv2.imwrite(image_path, img)
    
def redact_text_with_image(image_path, text, blur=False):
    """Redact or blur sensitive text in an image."""
    sensitive_data = get_sensitive_data(text)

    # Get bounding boxes of sensitive data
    for data in sensitive_data.keys():
        # Detect the bounding boxes for sensitive data using Tesseract
        boxes = pytesseract.image_to_boxes(Image.open(image_path))
        
        for box in boxes.splitlines():
            b = box.split()
            if b[0] == data:
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

                if blur:
                    apply_blur_on_image(image_path, (x, y, w, h))
                else:
                    # Redact with black box
                    img = cv2.imread(image_path)
                    img[y:h, x:w] = COLOR_MASK
                    cv2.imwrite(image_path, img)


def redact_text_with_pymupdf(doc, blur=False):
    """Redact or blur sensitive text in a PDF."""
    for page in doc:
        page.wrap_contents()
        text = page.get_text("text")
        
        # Detect sensitive info
        sensitive_data = get_sensitive_data(text)

        for data in sensitive_data.keys():
            raw_areas = page.search_for(data)

            for area in raw_areas:
                extracted_text = page.get_text("text", clip=area).strip()
                if extracted_text == data:
                    if blur:
                        apply_blur(page, area)
                    else:
                        page.add_redact_annot(area, fill=color_map[COLOR_MASK])

        page.apply_redactions()

    return doc



def process_image(image_path, output_image_path, blur=False):
    """Complete process for images: Detect sensitive data, redact/blur, and save."""
    text = extract_text_from_image(image_path)
    
    # ✅ Redact or blur sensitive data in the image
    redact_text_with_image(image_path, text, blur=blur)
    
    # ✅ Save the processed image
    cv2.imwrite(output_image_path, cv2.imread(image_path))

    return output_image_path, text


def process_pdf(pdf_path, output_pdf, blur=False):
    """Complete process: Extract text, detect sensitive info, tokenize, redact/blur, and save."""
    text, doc = extract_text_from_pdf(pdf_path)

    # ✅ Redact or blur sensitive data in PDF
    redacted_doc = redact_text_with_pymupdf(doc, blur=blur)

    # ✅ Save the redacted/blurred PDF
    print(f"Saving redacted PDF to: {output_pdf}")

    redacted_doc.save(output_pdf)

    return output_pdf, text 

def get_token_map(text):
    """
    Get a mapping of detected entities with their original name and entity type.
    """
    analysis_results = analyzer.analyze(text=text, entities=None, language="en")
    
    token_map = {}
    for result in analysis_results:
        entity_text = text[result.start:result.end]  # Extract original entity text
        entity_label = result.entity_type  # Get entity type
        
        token_map[entity_text] = entity_label

    return token_map

@app.route("/process", methods=["POST"])
def process_file_endpoint():
    """Upload & process PDFs and images."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    # Secure filename and save original file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    output_filename = "processed_" + filename
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)

    # Check file type & process accordingly
    if filename.lower().endswith('.pdf'):
        process_pdf(input_path, output_path, blur=False)  # Save processed PDF
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(input_path, output_path, blur=False)  # Save processed image
    else:
        return "Unsupported file format", 400

    # Redirect to preview page
    return redirect(url_for("process_file", filename=output_filename))



@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            return redirect(url_for("preview_file", filename=filename))

    return render_template("upload.html")

@app.route("/preview/<filename>")
def preview_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return render_template("preview.html", filename=filename)
@app.route("/process/<filename>")
def process_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    print(f"File path for upload: {file_path}")  # Debugging file path

    # Check if the file exists
    if not os.path.exists(file_path):
        return f"File {filename} not found in upload directory", 404

    if filename.lower().endswith(".pdf"):
        output_filename = "processed_" + filename  # Example: "processed_filename.pdf"
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        print(f"Saving processed PDF to: {output_path}")  # Debugging output path
        _, extracted_text = process_pdf(file_path, output_path)
    else:
        output_filename = "processed_" + filename  # Example: "processed_filename.png"
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        print(f"Saving processed image to: {output_path}")  # Debugging output path
        _, extracted_text = process_image(file_path, output_path)

    sensitive_data = get_sensitive_data(extracted_text)
    store_sensitive_data_firestore(sensitive_data)

    # Return the result page with the processed filename
    # print(sensitive_data)
    return render_template(
        "result.html",
        filename=output_filename,
        extracted_text=extracted_text,
        sensitive_data=sensitive_data,
    )

from flask import send_from_directory
@app.route("/download/<filename>")
def download_file(filename):
    """Allow user to download processed file."""
    file_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)  # FIXED
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404


if __name__ == "__main__":
    try:
        app.config['DEBUG'] = True
        app.run(debug=True, port=5001, host='0.0.0.0')
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...") # This keeps the window open

# if __name__ == "__main__":
#     # ✅ Sample text to test the model
#     file_path = r"D:\CodeFest(tokenization)\Codefest_Token\Codefest_Token\image.png"
    
#     # Process PDF or image based on the file type
#     if file_path.lower().endswith(".pdf"):
#         print(f"\n🔹 Processing PDF: {file_path}")
#         processed_pdf, extracted_text = process_pdf(file_path, "processed_output.pdf")

#     elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
#         print(f"\n🔹 Processing Image: {file_path}")
#         processed_image, extracted_text = process_image(file_path, "processed_output.png")

#     else:
#         print("❌ Unsupported file type! Please provide a PDF or image.")
#         exit(1)

#     # ✅ Get token-wise entity mapping
#     entity_map = get_token_map(extracted_text)

#     # ✅ Get sensitive data with secure tokens and confidence scores
#     sensitive_data = get_sensitive_data(extracted_text)

#     # ✅ Print the results
#     print("\n📌 Entity Map with Secure Tokens and Confidence Scores:")
#     if sensitive_data:
#         for token, details in sensitive_data.items():
#             # Extract the safe token and confidence score
#             safe_token = details.get("safe_token")
#             confidence_score = details.get("confidence_score")
#             entity_type = details.get("entity")
#             if entity_type=="IN_PAN" and confidence_score<=0.7:
#                 continue
#             # Print the token, entity, generated secure token, and confidence score
#             print(f"Token: {token} -> Entity: {entity_type} -> Safe Token: {safe_token} -> Confidence Score: {confidence_score:.2f}")
#     else:
#         print("No sensitive entities found in the document.")
