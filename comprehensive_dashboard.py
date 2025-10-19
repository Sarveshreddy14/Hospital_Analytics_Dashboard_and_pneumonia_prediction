import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from PIL import Image
import joblib
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics from JSON files"""
    metrics = {}
    
    # MIMIC-III metrics
    mimic_path = Path("processed/mimic/evaluation_metrics.json")
    if mimic_path.exists():
        with open(mimic_path, 'r') as f:
            metrics['mimic'] = json.load(f)
    
    # X-ray metrics
    xray_path = Path("model_store/cxr_test_metrics.json")
    if xray_path.exists():
        with open(xray_path, 'r') as f:
            metrics['xray'] = json.load(f)
    
    return metrics

@st.cache_data
def load_association_rules():
    """Load association rules from CSV"""
    rules_path = Path("processed/mimic/assoc_rules_dx.csv")
    if rules_path.exists():
        return pd.read_csv(rules_path)
    return None

@st.cache_data
def load_physionet_data():
    """Load PhysioNet processed data"""
    physionet_path = Path("processed/physionet/physionet_processed.csv")
    if physionet_path.exists():
        return pd.read_csv(physionet_path)
    return None

@st.cache_data
def load_feedback_data():
    """Load patient feedback data"""
    feedback_path = Path("processed/feedback/patient_feedback_ml.csv")
    if feedback_path.exists():
        return pd.read_csv(feedback_path)
    return None

@st.cache_data
def load_synthea_data():
    """Load Synthea synthetic data"""
    synthea_path = Path("synthetic_health_risk_prediction/Health_Risk_Dataset.csv")
    if synthea_path.exists():
        return pd.read_csv(synthea_path)
    return None

@st.cache_resource
def load_xray_model():
    """Load the trained X-ray model"""
    model_path = Path("model_store/cxr_effb0_best.keras")
    if model_path.exists():
        try:
            return tf.keras.models.load_model(str(model_path))
        except Exception as e:
            st.error(f"Error loading X-ray model: {e}")
            return None
    return None

def preprocess_xray_image(image, target_size=(224, 224)):
    """Preprocess X-ray image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_pneumonia(image):
    """Predict pneumonia from X-ray image"""
    model = load_xray_model()
    if model is None:
        return None, "Model not available"
    
    try:
        processed_image = preprocess_xray_image(image)
        prediction = model.predict(processed_image, verbose=0)
        probability = prediction[0][0]
        predicted_class = "PNEUMONIA" if probability > 0.5 else "NORMAL"
        confidence = probability if predicted_class == "PNEUMONIA" else 1 - probability
        return predicted_class, confidence, probability
    except Exception as e:
        return None, f"Prediction error: {e}"

# Sidebar navigation
st.sidebar.title("🏥 Hospital Analytics")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Home", "Physionet Vitals", "Pneumonia Prediction", "Patient Feedback", "Synthea Detection"]
)

# Load all data
metrics = load_evaluation_metrics()
association_rules = load_association_rules()
physionet_data = load_physionet_data()
feedback_data = load_feedback_data()
synthea_data = load_synthea_data()

# Home Page
if page == "Home":
    st.markdown('<h1 class="main-header">🏥 Hospital Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    This comprehensive hospital analytics dashboard integrates multiple machine learning models and datasets to provide insights across various healthcare domains:
    
    ### 🩺 **MIMIC-III Clinical Data Analysis**
    - **Mortality Prediction**: Binary classification using clinical features
    - **Length of Stay Prediction**: Regression analysis for hospital resource planning
    - **Patient Clustering**: K-means clustering for patient stratification
    - **Association Rule Mining**: Discovery of clinical pattern relationships
    
    ### 💓 **PhysioNet Vital Signs Analysis**
    - **Sepsis Prediction**: Time-series analysis of vital signs
    - **Feature Engineering**: Rolling statistics and trend analysis
    - **Missing Value Imputation**: Advanced imputation strategies
    
    ### 🫁 **Chest X-ray Pneumonia Detection**
    - **Deep Learning Model**: EfficientNetB0-based CNN
    - **Real-time Prediction**: Upload and classify chest X-ray images
    - **High Accuracy**: 83% accuracy with 93% recall for pneumonia detection
    
    ### 💬 **Patient Feedback Sentiment Analysis**
    - **NLP Processing**: Text preprocessing and feature extraction
    - **Sentiment Classification**: Positive/negative feedback analysis
    - **Rating Prediction**: 1-5 scale rating prediction
    
    ### 🧬 **Synthetic Health Risk Detection**
    - **Synthea Dataset**: Synthetic patient data analysis
    - **Risk Stratification**: Multi-class health risk prediction
    - **Feature Importance**: Understanding key risk factors
    
    ### 📊 **Key Metrics Tracked**
    - **Classification**: Accuracy, F1-score, ROC-AUC
    - **Regression**: RMSE, MAE, R²
    - **Clustering**: Silhouette, Calinski-Harabasz scores
    - **Associations**: Support, Confidence, Lift
    - **Imaging**: Precision, Recall, AUC
    - **Sentiment**: Precision/Recall, MCC
    """)
    
    # Display overall metrics summary
    if metrics:
        st.markdown("### 📈 Overall Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'mimic' in metrics:
                st.metric("MIMIC Classification F1", f"{metrics['mimic']['classification']['f1_score']:.3f}")
        
        with col2:
            if 'xray' in metrics:
                st.metric("X-ray Accuracy", f"{metrics['xray']['accuracy']:.3f}")
        
        with col3:
            if 'mimic' in metrics:
                st.metric("LOS R² Score", f"{metrics['mimic']['regression']['r2']:.3f}")
        
        with col4:
            if 'mimic' in metrics:
                st.metric("Clustering Silhouette", f"{metrics['mimic']['clustering']['silhouette']:.3f}")

# Physionet Vitals Page
elif page == "Physionet Vitals":
    st.markdown('<h1 class="section-header">💓 PhysioNet Vital Signs Analysis</h1>', unsafe_allow_html=True)
    
    if physionet_data is not None:
        st.markdown("### Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(physionet_data):,}")
        
        with col2:
            st.metric("Features", f"{physionet_data.shape[1] - 1}")
        
        with col3:
            sepsis_rate = physionet_data['SepsisLabel'].mean() * 100
            st.metric("Sepsis Rate", f"{sepsis_rate:.2f}%")
        
        # Display sample data
        st.markdown("### Sample Data")
        st.dataframe(physionet_data.head(10))
        
        # Feature distribution
        st.markdown("### Key Vital Signs Distribution")
        
        vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
        available_vitals = [col for col in vital_signs if col in physionet_data.columns]
        
        if available_vitals:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, vital in enumerate(available_vitals[:6]):
                if i < len(axes):
                    axes[i].hist(physionet_data[vital].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'{vital} Distribution')
                    axes[i].set_xlabel(vital)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Sepsis analysis
        st.markdown("### Sepsis vs Non-Sepsis Analysis")
        
        sepsis_data = physionet_data[physionet_data['SepsisLabel'] == 1]
        normal_data = physionet_data[physionet_data['SepsisLabel'] == 0]
        
        if not sepsis_data.empty and not normal_data.empty:
            comparison_data = []
            for vital in available_vitals[:6]:
                comparison_data.append({
                    'Vital Sign': vital,
                    'Sepsis Mean': sepsis_data[vital].mean(),
                    'Normal Mean': normal_data[vital].mean(),
                    'Difference': sepsis_data[vital].mean() - normal_data[vital].mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(comparison_df))
            width = 0.35
            
            ax.bar(x - width/2, comparison_df['Normal Mean'], width, label='Normal', alpha=0.8)
            ax.bar(x + width/2, comparison_df['Sepsis Mean'], width, label='Sepsis', alpha=0.8)
            
            ax.set_xlabel('Vital Signs')
            ax.set_ylabel('Mean Value')
            ax.set_title('Vital Signs: Sepsis vs Normal')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Vital Sign'])
            ax.legend()
            
            st.pyplot(fig)
    
    else:
        st.warning("PhysioNet data not found. Please run the preprocessing notebook first.")

# Pneumonia Prediction Page
elif page == "Pneumonia Prediction":
    st.markdown('<h1 class="section-header">🫁 Chest X-ray Pneumonia Detection</h1>', unsafe_allow_html=True)
    
    # Display model metrics
    if 'xray' in metrics:
        st.markdown("### Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['xray']['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics['xray']['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics['xray']['recall']:.3f}")
        
        with col4:
            st.metric("ROC AUC", f"{metrics['xray']['roc_auc']:.3f}")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm = np.array(metrics['xray']['confusion_matrix'])
        classes = metrics['xray']['classes']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # Image upload and prediction
    st.markdown("### Upload Chest X-ray for Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image to predict if it shows pneumonia or normal condition"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Make prediction
        if st.button("Predict Pneumonia"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probability = predict_pneumonia(image)
                
                if predicted_class is not None:
                    st.success(f"Prediction: {predicted_class}")
                    st.info(f"Confidence: {confidence:.2%}")
                    
                    # Display probability bar
                    prob_normal = 1 - probability
                    prob_pneumonia = probability
                    
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.barh(['Normal', 'Pneumonia'], [prob_normal, prob_pneumonia], 
                           color=['green', 'red'], alpha=0.7)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    
                    # Add probability labels
                    ax.text(prob_normal/2, 0, f'{prob_normal:.2%}', 
                           ha='center', va='center', color='white', fontweight='bold')
                    ax.text(prob_pneumonia/2, 1, f'{prob_pneumonia:.2%}', 
                           ha='center', va='center', color='white', fontweight='bold')
                    
                    st.pyplot(fig)
                else:
                    st.error(f"Prediction failed: {confidence}")

# Patient Feedback Page
elif page == "Patient Feedback":
    st.markdown('<h1 class="section-header">💬 Patient Feedback Analysis</h1>', unsafe_allow_html=True)
    
    if feedback_data is not None:
        st.markdown("### Dataset Overview")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", len(feedback_data))
        
        with col2:
            if 'label' in feedback_data.columns:
                positive_rate = feedback_data['label'].mean() * 100
                st.metric("Positive Sentiment", f"{positive_rate:.1f}%")
        
        with col3:
            if 'rating' in feedback_data.columns:
                avg_rating = feedback_data['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}")
        
        # Sentiment distribution
        if 'label' in feedback_data.columns:
            st.markdown("### Sentiment Distribution")
            
            sentiment_counts = feedback_data['label'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colors = ['#ff6b6b', '#4ecdc4']
            labels = ['Negative', 'Positive']
            
            ax.pie(sentiment_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
        
        # Rating distribution
        if 'rating' in feedback_data.columns:
            st.markdown("### Rating Distribution")
            
            rating_counts = feedback_data['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
            ax.set_xlabel('Rating (1-5)')
            ax.set_ylabel('Count')
            ax.set_title('Rating Distribution')
            ax.set_xticks(range(1, 6))
            
            st.pyplot(fig)
        
        # Text features analysis
        text_features = ['word_count', 'char_count', 'avg_word_length', 'sentence_count']
        available_text_features = [col for col in text_features if col in feedback_data.columns]
        
        if available_text_features:
            st.markdown("### Text Features Analysis")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(available_text_features):
                if i < len(axes):
                    axes[i].hist(feedback_data[feature].dropna(), bins=30, alpha=0.7, color='lightcoral')
                    axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                    axes[i].set_xlabel(feature.replace("_", " "))
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Sample feedback
        st.markdown("### Sample Feedback Data")
        display_cols = ['label', 'rating'] + available_text_features
        available_display_cols = [col for col in display_cols if col in feedback_data.columns]
        
        if available_display_cols:
            st.dataframe(feedback_data[available_display_cols].head(10))
    
    else:
        st.warning("Patient feedback data not found. Please run the preprocessing notebook first.")

# Synthea Detection Page
elif page == "Synthea Detection":
    st.markdown('<h1 class="section-header">🧬 Synthetic Health Risk Detection</h1>', unsafe_allow_html=True)
    
    if synthea_data is not None:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(synthea_data))
        
        with col2:
            st.metric("Features", synthea_data.shape[1])
        
        with col3:
            if 'Health_Risk' in synthea_data.columns:
                risk_dist = synthea_data['Health_Risk'].value_counts()
                st.metric("Most Common Risk", risk_dist.index[0])
        
        # Display sample data
        st.markdown("### Sample Data")
        st.dataframe(synthea_data.head(10))
        
        # Health risk distribution
        if 'Health_Risk' in synthea_data.columns:
            st.markdown("### Health Risk Distribution")
            
            risk_counts = synthea_data['Health_Risk'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(range(len(risk_counts)), risk_counts.values, color='lightgreen', alpha=0.7)
            ax.set_xlabel('Health Risk Level')
            ax.set_ylabel('Count')
            ax.set_title('Health Risk Distribution')
            ax.set_xticks(range(len(risk_counts)))
            ax.set_xticklabels(risk_counts.index, rotation=45)
            
            st.pyplot(fig)
        
        # Feature analysis
        numeric_cols = synthea_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Health_Risk' in numeric_cols:
            numeric_cols.remove('Health_Risk')
        
        if numeric_cols:
            st.markdown("### Feature Distributions")
            
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
                if i < len(axes):
                    axes[i].hist(synthea_data[col].dropna(), bins=30, alpha=0.7, color='lightblue')
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.markdown("### Feature Correlation Matrix")
            
            corr_matrix = synthea_data[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix')
            
            st.pyplot(fig)
    
    else:
        st.warning("Synthea data not found. Please check the data path.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🏥 Hospital Analytics Dashboard | Powered by Streamlit</p>
        <p>Comprehensive ML models for healthcare insights</p>
    </div>
    """, 
    unsafe_allow_html=True
)
