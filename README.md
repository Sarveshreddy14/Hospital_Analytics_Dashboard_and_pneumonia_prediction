# 🏥 Hospital Analytics Dashboard

A comprehensive machine learning project for healthcare analytics, featuring multiple models for clinical prediction, medical imaging, and patient feedback analysis.

## 🚀 Project Overview

This project integrates multiple healthcare datasets and machine learning models to provide insights across various medical domains:

- **MIMIC-III Clinical Data**: Mortality prediction, length of stay forecasting, patient clustering
- **PhysioNet Vital Signs**: Sepsis prediction using time-series analysis
- **Chest X-ray Analysis**: Pneumonia detection using deep learning
- **Patient Feedback**: Sentiment analysis and rating prediction
- **Synthetic Health Data**: Risk stratification using Synthea dataset

## 📊 Key Features

### 🩺 Clinical Predictions
- **Mortality Classification**: Binary classification with 80%+ F1-score
- **Length of Stay Regression**: MAE < baseline predictions
- **Patient Clustering**: K-means with clinical interpretability
- **Association Rules**: Medical pattern discovery

### 💓 Vital Signs Analysis
- **Sepsis Detection**: Time-series feature engineering
- **Missing Value Imputation**: Advanced imputation strategies
- **Rolling Statistics**: 6-hour window analysis

### 🫁 Medical Imaging
- **Pneumonia Detection**: EfficientNetB0 CNN model
- **Real-time Prediction**: Upload and classify X-ray images
- **High Performance**: 83% accuracy, 93% recall

### 💬 Patient Feedback
- **Sentiment Analysis**: NLP processing with TF-IDF
- **Rating Prediction**: 1-5 scale classification
- **Text Features**: Word count, sentiment scoring

### 🧬 Synthetic Data
- **Health Risk Detection**: Multi-class risk stratification
- **Feature Correlation**: Understanding risk factors
- **Synthetic Validation**: Synthea dataset analysis

## 🛠️ Technology Stack

- **Python 3.11+**
- **Machine Learning**: Scikit-learn, TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Interface**: Streamlit, FastAPI
- **NLP**: NLTK, WordCloud
- **Association Mining**: MLxtend

## 📁 Project Structure

```
hospital_analytics/
├── 📊 Data Processing
│   ├── preprocessing.ipynb              # MIMIC-III preprocessing
│   ├── physionet_preprocessing.ipynb    # PhysioNet time-series
│   ├── patient_feedback_preprocessing.ipynb  # NLP preprocessing
│   └── synthetic_synthea.ipynb          # Synthetic data analysis
│
├── 🤖 Model Training
│   ├── mimic_3.ipynb                    # Clinical prediction models
│   ├── physionet_code.ipynb             # Sepsis prediction
│   ├── xray_chest.ipynb                 # Pneumonia detection
│   ├── biobert_transfer_learning.ipynb  # BERT for medical text
│   └── association_rules_mining.ipynb   # Pattern discovery
│
├── 🌐 Web Applications
│   ├── comprehensive_dashboard.py       # Main Streamlit dashboard
│   ├── streamlit_app.py                 # FastAPI integration
│   └── api/                             # FastAPI backend
│
├── 📈 Data & Models
│   ├── processed/                       # Processed datasets
│   ├── model_store/                     # Trained models
│   └── MIMIC-III/                       # Raw clinical data
│
└── 📚 Documentation
    ├── README.md                        # This file
    ├── README_DASHBOARD.md              # Dashboard guide
    └── requirements.txt                 # Dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hospital-analytics.git
cd hospital-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
# Option 1: Using the run script
python run_dashboard.py

# Option 2: Direct Streamlit
streamlit run comprehensive_dashboard.py
```

### 4. Access the Dashboard
Open your browser to: `http://localhost:8501`

## 📊 Performance Metrics

### Classification Models
- **Mortality Prediction**: F1-Score > 0.80
- **Pneumonia Detection**: Accuracy 83%, Recall 93%
- **Sentiment Analysis**: Precision/Recall > 0.85

### Regression Models
- **Length of Stay**: MAE < baseline, R² > 0.60
- **Health Risk**: Multi-class accuracy > 0.75

### Clustering
- **Patient Groups**: Silhouette > 0.15
- **Clinical Interpretability**: Meaningful clusters

### Association Rules
- **Medical Patterns**: Support > 0.01, Confidence > 0.5
- **Lift**: > 2.0 for significant associations

## 🔬 Datasets Used

1. **MIMIC-III**: Critical care database (MIT)
2. **PhysioNet**: Sepsis prediction challenge
3. **Chest X-ray**: Pneumonia detection dataset
4. **Patient Feedback**: Hospital experience data
5. **Synthea**: Synthetic health records

## 📈 Key Insights

- **Clinical Patterns**: Discovered 20+ significant association rules
- **Risk Factors**: Identified key predictors for mortality and sepsis
- **Resource Planning**: Improved length of stay predictions
- **Patient Experience**: Analyzed sentiment patterns in feedback
- **Diagnostic Support**: High-accuracy pneumonia detection

## 🛡️ Data Privacy & Ethics

- All datasets used are publicly available or synthetic
- No real patient data stored in the repository
- Models trained on de-identified data only
- Ethical AI principles followed throughout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIMIC-III**: MIT Laboratory for Computational Physiology
- **PhysioNet**: National Institute of Biomedical Imaging and Bioengineering
- **Chest X-ray Dataset**: Paul Mooney, Daniel Kermany
- **Synthea**: The MITRE Corporation

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**Built with ❤️ for Healthcare Innovation**

*This project demonstrates the power of machine learning in healthcare analytics, combining multiple data sources and model types to provide comprehensive insights for medical professionals and researchers.*
