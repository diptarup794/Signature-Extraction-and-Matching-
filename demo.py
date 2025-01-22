import streamlit as st
import os
import fitz
from PIL import Image
import cv2
import numpy as np
import logging
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tempfile
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_WIDTH = 8.0833
PDF_HEIGHT = 11.6806
DPI = 300
TARGET_WIDTH = int(PDF_WIDTH * DPI)
TARGET_HEIGHT = int(PDF_HEIGHT * DPI)

TEMP_DIR = tempfile.mkdtemp()
os.makedirs(os.path.join(TEMP_DIR, 'signatures'), exist_ok=True)

SIGNATURES_CONFIG = [
    {
        "holder": "1st",
        "page": 5,
        "coordinates": [
            {"x": 6.0448, "y": 5.8352},
            {"x": 7.8044, "y": 5.8352},
            {"x": 7.8044, "y": 5.9772},
            {"x": 6.0448, "y": 5.9772},
        ],
        "output": "signature_1st_holder.png",
        "y_offset": 0.85
    },
    {
        "holder": "2nd",
        "page": 6,
        "coordinates": [
            {"x": 0.8566, "y": 10.6261},
            {"x": 1.6118, "y": 10.5348},
            {"x": 1.6523, "y": 10.8795},
            {"x": 0.8819, "y": 10.9961},
        ],
        "output": "signature_2nd_holder.png",
        "y_offset": 0.85
    },
    {
        "holder": "3rd",
        "page": 7,
        "coordinates": [
            {"x": 0.8358, "y": 4.5522},
            {"x": 1.6614, "y": 4.3193},
            {"x": 1.7272, "y": 4.6181},
            {"x": 0.9016, "y": 4.8307},
        ],
        "output": "signature_3rd_holder.png",
        "y_offset": 0.85
    }
]

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

def convert_pdf_to_images(pdf_file):
    try:
        pdf_path = os.path.join(TEMP_DIR, "temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        pdf_document = fitz.open(pdf_path)
        page_images = []
        
        with st.spinner('Converting PDF pages...'):
            progress_bar = st.progress(0)
            
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                zoom_x = TARGET_WIDTH / page.rect.width
                zoom_y = TARGET_HEIGHT / page.rect.height
                mat = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                
                image_path = os.path.join(TEMP_DIR, f"page_{page_number + 1}.png")
                img.save(image_path, "PNG")
                page_images.append(image_path)
                
                progress_bar.progress(float((page_number + 1) / len(pdf_document)))
        
        return page_images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        raise



def detect_signature(image_path):
    """Detect if a signature is present using edge detection"""
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Count non-zero pixels (edges)
        edge_pixels = np.count_nonzero(edges)
        
        # Calculate percentage of edge pixels
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        # Return True if edge percentage is above threshold (indicating signature presence)
        return edge_percentage > 0.5
    except Exception as e:
        st.error(f"Error detecting signature: {str(e)}")
        raise

def save_signature(image_path, coordinates, holder_type, y_offset=0.85):
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            x_scale = img_width / PDF_WIDTH
            y_scale = img_height / PDF_HEIGHT
            
            if holder_type == "1st":
                x_min = int(coordinates[0]["x"] * x_scale)
                y_max = int(coordinates[0]["y"] * y_scale)
                x_max = int(coordinates[2]["x"] * x_scale) - 50
                y_min = int((coordinates[2]["y"] - y_offset) * y_scale)
            elif holder_type == "2nd":
                x_min = int(coordinates[0]["x"] * x_scale) - 50
                y_min = int((coordinates[0]["y"] - y_offset) * y_scale) + 220
                x_max = int(coordinates[2]["x"] * x_scale) + 50
                y_max = int(coordinates[2]["y"] * y_scale) + 50
            else:
                x_min = int(coordinates[0]["x"] * x_scale) - 50
                y_min = int((coordinates[0]["y"] - y_offset) * y_scale) + 190
                x_max = int(coordinates[2]["x"] * x_scale) + 50
                y_max = int(coordinates[2]["y"] * y_scale) + 75

            if y_min > y_max:
                y_min, y_max = y_max, y_min

            cropped = img.crop((x_min, y_min, x_max, y_max))
            output_path = os.path.join(TEMP_DIR, 'signatures', f"signature_{holder_type}_holder.png")
            cropped.save(output_path, "PNG")
            
            # Check if signature is present
            has_signature = detect_signature(output_path)
            if not has_signature:
                os.remove(output_path)
                return None
                
            return output_path
    except Exception as e:
        st.error(f"Error saving signature: {str(e)}")
        raise

def load_and_preprocess_image(image_source, target_size=(224, 224)):
    """Load and preprocess image from various sources"""
    try:
        if isinstance(image_source, (str, os.PathLike)):
            # Load from file path
            img = cv2.imread(str(image_source))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_source, bytes):
            # Load from bytes
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_source, np.ndarray):
            # Already a numpy array
            img = image_source.copy()
            if len(img.shape) == 2:  # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Unsupported image source type")

        # Resize and preprocess
        img = cv2.resize(img, target_size)
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        raise

def extract_features(image_source):
    """Extract features from an image source"""
    try:
        preprocessed_img = load_and_preprocess_image(image_source)
        model = load_model()
        return model.predict(preprocessed_img, batch_size=1)
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        raise

def cosine_similarity(features1, features2):
    return float(np.dot(features1.flatten(), features2.flatten()) / (
        np.linalg.norm(features1) * np.linalg.norm(features2)
    ))

def compare_signatures(uploaded_file, extracted_signature_path):
    """Compare uploaded signature with extracted signature"""
    try:
        # Read and extract features from uploaded signature
        uploaded_bytes = uploaded_file.read()
        uploaded_features = extract_features(uploaded_bytes)
        
        # Reset file pointer for future reads
        uploaded_file.seek(0)
        
        # Extract features from extracted signature
        extracted_features = extract_features(extracted_signature_path)
        
        # Calculate similarity
        similarity = cosine_similarity(uploaded_features, extracted_features)
        return similarity
    except Exception as e:
        st.error(f"Error comparing signatures: {str(e)}")
        raise



def main():
    st.title("Signature Verification System")
    
    if 'signatures' not in st.session_state:
        st.session_state.signatures = {}
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
    
    if uploaded_file is not None and st.button("Process PDF"):
        try:
            st.session_state.signatures = {}  # Clear previous signatures
            page_images = convert_pdf_to_images(uploaded_file)
            
            with st.spinner("Extracting signatures..."):
                progress_bar = st.progress(0)
                
                for idx, sig_config in enumerate(SIGNATURES_CONFIG):
                    page_image = page_images[sig_config["page"] - 1]
                    sig_path = save_signature(
                        page_image,
                        sig_config["coordinates"],
                        sig_config["holder"],
                        sig_config["y_offset"]
                    )
                    
                    if sig_path:  # Only store if signature was detected
                        st.session_state.signatures[sig_config["holder"]] = sig_path
                    
                    progress_bar.progress(float((idx + 1) / len(SIGNATURES_CONFIG)))
            
            if st.session_state.signatures:
                st.success("Signatures extracted successfully!")
                st.session_state.pdf_processed = True
            else:
                st.error("No signatures were detected in the document.")
                st.session_state.pdf_processed = False
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state.pdf_processed = False
    
    # Display extracted signatures if available
    if st.session_state.signatures:
        st.subheader("Extracted Signatures")
        available_holders = list(st.session_state.signatures.keys())
        cols = st.columns(len(available_holders))
        for idx, holder in enumerate(available_holders):
            cols[idx].image(st.session_state.signatures[holder], 
                          caption=f"{holder} Holder's Signature")
        
        # Only show signature upload after PDF processing
        st.subheader("Compare Signatures")
        uploaded_signature = st.file_uploader("Upload Signature to Compare", 
                                            type=['png', 'jpg', 'jpeg'])
        
        if uploaded_signature:
            try:
                # Display uploaded signature
                st.image(uploaded_signature, caption="Uploaded Signature", width=200)
                
                # Compare with all extracted signatures
                results = []
                for holder in st.session_state.signatures.keys():
                    similarity = compare_signatures(uploaded_signature, 
                                                 st.session_state.signatures[holder])
                    results.append((holder, similarity))
                
                # Display results in a table
                results_df = {
                    "Holder": [],
                    "Match Percentage": [],
                    "Status": []
                }
                
                for holder, score in results:
                    results_df["Holder"].append(holder)
                    results_df["Match Percentage"].append(f"{score * 100:.2f}%")
                    results_df["Status"].append("Match" if score >= 0.9 else "No Match")
                
                st.table(results_df)
                
                # Find best match
                best_match = max(results, key=lambda x: x[1])
                if best_match[1] >= 0.9:
                    st.success(f"Best match: {best_match[0]} Holder with {best_match[1]*100:.2f}% similarity")
                else:
                    st.warning("No strong matches found in any extracted signatures")
                
            except Exception as e:
                st.error(f"Error comparing signatures: {str(e)}")

if __name__ == "__main__":
    main()