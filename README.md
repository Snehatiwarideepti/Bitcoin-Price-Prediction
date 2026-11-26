# Bitcoin Price Prediction

A machine learning project for predicting Bitcoin prices using historical market data and data-driven forecasting techniques.

## Overview

Bitcoin is a highly volatile digital asset whose price is influenced by market trends and historical price movements. This project explores historical Bitcoin data and applies machine learning techniques to build a price prediction pipeline.

The project includes data processing, model training, prediction generation, and output visualization.

## Project Structure

```text
Bitcoin-Price-Prediction/
│
├── data/
│   └── Dataset and processed data
│
├── models/
│   └── Trained machine learning models
│
├── outputs/
│   └── Generated predictions, visualizations, and results
│
├── train.py
│   └── Model training pipeline
│
├── requirements.txt
│   └── Python dependencies
│
└── IEEE_Bitcoin_Prediction_Paper (1).pdf
    └── Research paper/documentation
```

## Technologies Used

- Python
- Apache Spark
- PySpark
- Machine Learning
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Workflow

The project follows a machine learning pipeline:

1. **Data Collection**  
   Historical Bitcoin price data is used as the input dataset.

2. **Data Preprocessing**  
   The data is cleaned and prepared for machine learning.

3. **Feature Processing**  
   Relevant numerical features are prepared for model training.

4. **Model Training**  
   Machine learning models are trained using the processed historical data.

5. **Prediction**  
   The trained models generate Bitcoin price predictions.

6. **Evaluation & Visualization**  
   Model outputs and prediction results are analyzed using generated visualizations.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Snehatiwarideepti/Bitcoin-Price-Prediction.git
```

### 2. Navigate to the project

```bash
cd Bitcoin-Price-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training script

```bash
python train.py
```

> Depending on the local Spark/PySpark configuration, additional Apache Spark setup may be required.

## Results

The generated model files, prediction outputs, and visualizations are available in the `models/` and `outputs/` directories.

The repository also contains the associated IEEE research paper/documentation in PDF format.

## Research Component

This project is accompanied by an IEEE-style research paper discussing the approach used for Bitcoin price prediction.

**Paper:** `IEEE_Bitcoin_Prediction_Paper (1).pdf`

## Future Improvements

- Experiment with additional time-series forecasting techniques
- Incorporate more market and technical indicators
- Compare multiple machine learning models
- Improve hyperparameter optimization
- Develop a real-time prediction interface
- Evaluate the models on continuously updated Bitcoin data

## Disclaimer

This project is developed for educational and research purposes. Bitcoin price prediction is inherently uncertain due to market volatility, and model predictions should not be considered financial advice.

## Author

**Sneha Tiwari**

B.Tech Computer Science Engineering  
Data Science
