# 🚲 Bike Demand Prediction using LSTM

This project predicts the hourly demand for rental bikes in urban areas using historical and real-time data, including weather conditions and time-related features. The goal is to ensure a stable supply of rental bikes to minimize user waiting time and improve city mobility.

## 📌 Problem Statement

Bike-sharing systems must provide bikes on time and in the right quantity. This project uses a deep learning model (LSTM) to predict the hourly bike count needed to meet demand efficiently.

## 🧠 ML Model

We used a **Long Short-Term Memory (LSTM)** model, ideal for time-series forecasting, trained on features like:
- Hour of the day
- Weather conditions
- Working day / Holiday
- Temperature, humidity, windspeed


## ⚙️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/Santhosh944/bike-demand-prediction.git
   cd bike-demand-prediction
2. Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model:
bash
Copy
Edit
python train_lstm_model.py
4. Run the Flask app:
bash
Copy
Edit
python app.py
5.Visit: http://127.0.0.1:5000/

Results
The model achieved promising accuracy with low MAE and RMSE. Visualizations showed close alignment between actual and predicted bike demand.

References
UCI Bike Sharing Dataset
Articles on LSTM for time-series forecasting
Blogs and GitHub repos on bike demand prediction
