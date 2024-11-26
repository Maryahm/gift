import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained models with corrected paths
alzheimers_model = tf.keras.models.load_model('C:/Users/User/OneDrive/Desktop/project/alzhimers_model.h5')
nail_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/nail disease.h5")
lung_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/Lung disease prediciton.h5")
brain_model = tf.keras.models.load_model("C:/Users/User/OneDrive/Desktop/project/brain ct modeel.h5")

# Mapping disease type to model
models = {
    'Alzheimer’s': alzheimers_model,
    'Nail Disease': nail_model,
    'Lung Disease': lung_model,
    'Brain Disease': brain_model  # Ensure spelling matches in both the dictionary and selectbox
}

# Set up the app title
st.title("Disease Prediction App")

# Disease type selection
disease_type = st.selectbox("Select the disease type for prediction:", ["Alzheimer’s", "Nail Disease", "Lung Disease", "Brain Disease"])

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
        prediction = model.predict(image)

        # Step 3: Display the prediction result
        st.write("Prediction Result:")
        if disease_type == "Alzheimer’s":
            categories = ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
            st.write(f"Dementia Level: {categories[np.argmax(prediction)]}")
        elif disease_type == "Nail Disease":
            categories = ["Healthy", "Diseased"]
            st.write(f"Nail Condition: {categories[np.argmax(prediction)]}")
        elif disease_type == "Lung Disease":
            categories = ["Healthy", "Diseased"]
            st.write(f"Lung Condition: {categories[np.argmax(prediction)]}")
        elif disease_type == "Brain Disease":
            categories = ["Healthy", "Diseased"]
            st.write(f"Brain Condition: {categories[np.argmax(prediction)]}")
    else:
        st.write("Please upload an image to make a prediction.")

