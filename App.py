import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained models with corrected paths
alzheimers_model = tf.keras.models.load_model('C:/Users/User/OneDrive/Desktop/project/alzhimers_model.h5')
nail_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/nail disease.h5")
lung_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/Lung disease prediciton.h5")
brain_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/brain ct modeel.h5")

# Placeholder for edema model (add model later)
# edema_model = tf.keras.models.load_model("path/to/edema_model.h5")

# Mapping disease type to model
models = {
    'Alzheimer’s': alzheimers_model,
    'Nail Disease': nail_model,
    'Lung Disease': lung_model,
    'Brain Disease': brain_model,
    'Edema': None  # Placeholder for edema model
}

# Class names for each model
class_names = {
    'Alzheimer’s': ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"],
    'Nail Disease': ['Healthy', 'Onychomycosis', 'Psoriasis'],
    'Lung Disease': ['Lung Opacity', 'Normal', 'Viral Pneumonia'],
    'Brain Disease': ['Aneurysm', 'Cancer', 'Tumor'],
    'Edema': ['Edema', 'No Edema']  # Add for edema when model is available
}

# Set up the app title with customizable font and color
st.title("Disease Prediction App")

# Optional custom CSS styling for fonts and colors
st.markdown(
    """
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        color: #5a2d90;  /* Purple color */
        font-size: 40px;
    }
    .sidebar .sidebar-content {
        background-color: #f1f1f1;
    }
    </style>
    """, unsafe_allow_html=True)

# Disease type selection
disease_type = st.selectbox("Select the disease type for prediction:", ["Alzheimer’s", "Nail Disease", "Lung Disease", "Brain Disease", "Edema"])

# Image upload option
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Prediction button
if st.button("Predict"):
    if uploaded_image is not None:
        # Step 1: Preprocess the image
        image = Image.open(uploaded_image).convert("RGB")
        image = image.resize((224, 224))  # Resize as per model input requirements
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Step 2: Predict using the selected model
        model = models[disease_type]
        if model is not None:
            prediction = model.predict(image)
            
            # Step 3: Display the prediction result
            st.write("Prediction Result:")
            categories = class_names[disease_type]
            st.write(f"{disease_type} Condition: {categories[np.argmax(prediction)]}")
        else:
            st.write("Model for Edema prediction not available yet.")
    else:
        st.write("Please upload an image to make a prediction.")
