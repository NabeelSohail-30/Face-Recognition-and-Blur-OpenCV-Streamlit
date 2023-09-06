# Import necessary libraries
import streamlit as st
import cv2
import numpy as np


# Function to detect faces and blur them
def detect_and_blur_faces(image):
    # Load a pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur the detected faces
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y + h, x:x + w] = face

    return image


# Create a Streamlit app
def main():
    st.title("Face Detection and Blur App")

    # Upload an image using Streamlit
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Detect and blur faces
        blurred_image = detect_and_blur_faces(image)

        # Display the blurred image
        st.image(blurred_image, caption='Blurred Image', use_column_width=True)


if __name__ == '__main__':
    main()
