# Signature Verification System

This project is a Signature Verification System built using Streamlit, aimed at extracting signatures from PDF documents and comparing them with uploaded signature images for verification. The system utilizes the ResNet50 model for feature extraction and employs cosine similarity to compare the features of the extracted and uploaded signatures.

## Features
- **PDF to Image Conversion**: Converts PDF pages to high-resolution images for signature extraction.
- **Signature Extraction**: Crops and saves signatures from specific positions in the PDF.
- **Signature Comparison**: Compares the extracted signatures with uploaded images using deep learning-based feature extraction.
- **Cosine Similarity**: Measures the similarity between two feature sets to determine a match.

## Technologies Used
- **Streamlit**: For creating the interactive web application.
- **TensorFlow Keras**: To load the pre-trained ResNet50 model for feature extraction.
- **PyMuPDF (Fitz)**: To handle PDF document processing.
- **Pillow (PIL)**: For image processing and manipulation.
- **OpenCV**: To preprocess and manipulate images.
- **NumPy**: For numerical computations.


