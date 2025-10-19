# 🏥 Hospital Analytics Dashboard

A comprehensive Streamlit dashboard integrating multiple machine learning models and healthcare datasets for advanced analytics and predictions.

## 🚀 Quick Start

### Option 1: Using the run script (Recommended)
```bash
python run_dashboard.py
```

### Option 2: Direct Streamlit command
```bash
pip install -r requirements.txt
streamlit run comprehensive_dashboard.py
```

The dashboard will be available at: http://localhost:8501

## 📊 Dashboard Sections

### 1. **Home** 🏠
- Project overview and introduction
- Overall performance metrics summary
- Navigation to all subsections

### 2. **Physionet Vitals** 💓
- Vital signs analysis from PhysioNet dataset
- Sepsis prediction metrics
- Feature distributions and comparisons
- Time-series analysis results

### 3. **Pneumonia Prediction** 🫁
- Chest X-ray pneumonia detection
- Model performance metrics (Accuracy, Precision, Recall, ROC-AUC)
- Real-time image upload and prediction
- Confusion matrix visualization

### 4. **Patient Feedback** 💬
- Sentiment analysis of patient feedback
- Rating distribution analysis
- Text feature analysis
- NLP processing results

### 5. **Synthea Detection** 🧬
- Synthetic health risk detection
- Feature correlation analysis
- Risk stratification visualization
- Synthetic data insights

## 📈 Metrics Tracked

### Classification Metrics
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Clustering Metrics
- **Silhouette Score**: Measure of cluster quality
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion

### Association Rules
- **Support**: Frequency of rule occurrence
- **Confidence**: Conditional probability
- **Lift**: Measure of rule strength

### Imaging Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC**: Area under the curve

### Sentiment Metrics
- **Precision/Recall**: For sentiment classification
- **MCC**: Matthews Correlation Coefficient

## 🗂️ Required Data Files

The dashboard expects the following processed data files:

```
processed/
├── mimic/
│   ├── evaluation_metrics.json
│   ├── assoc_rules_dx.csv
│   └── mimic_features.parquet
├── physionet/
│   └── physionet_processed.csv
└── feedback/
    ├── patient_feedback_ml.csv
    └── patient_feedback_transformer.csv

model_store/
├── cxr_effb0_best.keras
├── cxr_test_metrics.json
├── mimic_los_rf.joblib
└── tfidf_vectorizer.pkl

synthetic_health_risk_prediction/
└── Health_Risk_Dataset.csv
```

## 🔧 Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **TensorFlow**: Deep learning models
- **Scikit-learn**: Machine learning utilities
- **Pillow**: Image processing
- **MLxtend**: Association rule mining
- **NLTK**: Natural language processing

## 🚨 Troubleshooting

### Common Issues

1. **Missing data files**: Run the preprocessing notebooks first
2. **Model loading errors**: Ensure model files are in the correct directories
3. **Import errors**: Install all dependencies with `pip install -r requirements.txt`
4. **Memory issues**: Reduce batch sizes or sample sizes in preprocessing

### Data Preparation

Before running the dashboard, ensure you've executed:
1. `preprocessing.ipynb` - For MIMIC-III data
2. `physionet_preprocessing.ipynb` - For PhysioNet data
3. `patient_feedback_preprocessing.ipynb` - For feedback data
4. `xray_chest.ipynb` - For X-ray model training
5. `synthetic_synthea.ipynb` - For Synthea data

## 📱 Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Predictions**: Upload images for instant analysis
- **Interactive Visualizations**: Hover, zoom, and explore data
- **Comprehensive Metrics**: All evaluation metrics in one place
- **Easy Navigation**: Sidebar navigation between sections
- **Data Export**: Download processed data and results

## 🔮 Future Enhancements

- Real-time data streaming
- Advanced model comparison
- Automated retraining pipelines
- API integration for external data sources
- Advanced visualization options
- Export capabilities for reports

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all data files are present
3. Ensure all dependencies are installed
4. Check the console for error messages

---

**Built with ❤️ for Healthcare Analytics**
