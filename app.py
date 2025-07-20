# ====================================
# PLANT CLASSIFICATION STREAMLIT APP
# ====================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
from PIL import Image
import io
import os
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Plant Species Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_plant_model():
    """Load the trained plant classification model and metadata"""
    try:
        # Update these paths based on where you saved your model
        model_path = 'plant_classifier_model.h5'
        class_names_path = 'class_names.pkl'
        idx_to_class_path = 'idx_to_class.pkl'
        
        # Load model
        model = load_model(model_path)
        
        # Load class information
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        
        with open(idx_to_class_path, 'rb') as f:
            idx_to_class = pickle.load(f)
        
        return model, class_names, idx_to_class
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image for prediction"""
    try:
        # Resize image
        img = image.resize(target_size)
        # Convert to array
        img_array = img_to_array(img)
        # Normalize pixel values
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_plant(model, processed_image, class_names):
    """Make prediction on the processed image"""
    try:
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_idx:
            top_5_predictions.append({
                'Plant Species': class_names[idx],
                'Confidence': float(predictions[0][idx])
            })
        
        return predicted_class, confidence, top_5_predictions
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def create_confidence_chart(top_predictions):
    """Create a confidence chart for top predictions"""
    df = pd.DataFrame(top_predictions)
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Plant Species',
        orientation='h',
        title='Top 5 Prediction Confidence Scores',
        labels={'Confidence': 'Confidence Score', 'Plant Species': 'Plant Species'},
        color='Confidence',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def main():
    # App title and description
    st.title("🌱 Plant Species Classification")
    st.markdown("### Upload a plant image to identify the species")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model trained on plant species images 
        to classify and identify different plant species.
        
        **How to use:**
        1. Upload a clear image of a plant
        2. Wait for the model to process
        3. View the predicted species and confidence
        
        **Supported formats:** PNG, JPG, JPEG
        **Max file size:** 200MB
        """)
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Architecture:** VGG16 Transfer Learning
        - **Training Data:** Custom plant species dataset
        """)

    # Load model
    with st.spinner("Loading model..."):
        model, class_names, idx_to_class = load_plant_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if model files are available.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Can classify {len(class_names)} plant species.")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a plant for species identification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show image details
            st.write("**Image Details:**")
            st.write(f"- Filename: {uploaded_file.name}")
            st.write(f"- Size: {image.size}")
            st.write(f"- Format: {image.format}")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            
            # Prediction button
            if st.button("🚀 Classify Plant", type="primary", use_container_width=True):
                
                with st.spinner("Analyzing image... Please wait"):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        predicted_class, confidence, top_predictions = predict_plant(
                            model, processed_image, class_names
                        )
                        
                        if predicted_class is not None:
                            # Display main prediction
                            st.success("✅ Classification Complete!")
                            
                            # Main result
                            st.metric(
                                label="🌿 Predicted Plant Species",
                                value=predicted_class,
                                delta=f"{confidence:.2%} confidence"
                            )
                            
                            # Confidence level indicator
                            if confidence > 0.8:
                                st.success("🎯 High Confidence Prediction")
                            elif confidence > 0.6:
                                st.warning("⚠️ Moderate Confidence Prediction")
                            else:
                                st.error("❗ Low Confidence - Consider uploading a clearer image")
                            
                            # Store prediction in session state for history
                            if 'predictions' not in st.session_state:
                                st.session_state.predictions = []
                            
                            st.session_state.predictions.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': uploaded_file.name,
                                'predicted_class': predicted_class,
                                'confidence': confidence
                            })
        else:
            st.info("👆 Please upload an image to get started")
    
    # Display detailed results if prediction was made
    if uploaded_file is not None and st.session_state.get('predictions'):
        st.header("📈 Detailed Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🏆 Top Predictions", "📊 Confidence Chart", "📝 Prediction History"])
        
        with tab1:
            if 'top_predictions' in locals():
                st.subheader("Top 5 Most Likely Species")
                
                # Display top predictions as a nice table
                df_predictions = pd.DataFrame(top_predictions)
                df_predictions['Confidence'] = df_predictions['Confidence'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    df_predictions,
                    use_container_width=True,
                    hide_index=True
                )
        
        with tab2:
            if 'top_predictions' in locals():
                # Create and display confidence chart
                fig = create_confidence_chart(top_predictions)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Recent Predictions")
            if st.session_state.predictions:
                history_df = pd.DataFrame(st.session_state.predictions)
                history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.predictions = []
                    st.rerun()
            else:
                st.info("No predictions yet. Upload and classify some plants!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and TensorFlow | "
        "🌱 Plant Species Classification Model"
    )

if __name__ == "__main__":
    main()
