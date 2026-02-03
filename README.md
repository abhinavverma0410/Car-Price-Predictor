# 🚗 Car Price Prediction System

A data-driven Machine Learning application that predicts the market value of used cars using a trained dataset and supports cross-platform valuation through intelligent web scraping. The system is deployed as an interactive Plotly Dash web application.

---

## 📌 Project Overview

This project predicts the resale value of used cars based on patterns learned from a real-world dataset sourced from eBay listings. In addition to manual input, the application can extract vehicle details from multiple car listing websites using web scraping, enabling price prediction across different platforms.

---

## 🖥️ Application UI Preview
![Car Price Predictor UI](assets/carPricePredictor.png)

---

## 📊 Dataset

The model is trained on a cleaned version of the **eBay Used Car Sales dataset** sourced from Kaggle.

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/sijovm/used-cars-data-from-ebay-kleinanzeigen

The dataset contains historical used-car listings with:<br>
**Features:** _DateCrawled_ | _VehicleType_ | _RegistrationYear_ | _Gearbox_ | _Power_ | _Model_ | _Kilometer_ | _RegistrationMonth_ | _FuelType_ | _Brand_ | _Repaired_ | _DateCreated_ | _NumberOfPictures_ | _PostalCode_ | _LastSeen_<br>
**Target:** _Price (EUR)_

The raw dataset is cleaned and processed before being used for supervised machine learning.

---

## 🧠 Machine Learning Pipeline

- Custom **CarAgeTransformer** to compute vehicle age from registration data  
- Feature scaling and categorical encoding  
- **Weighted Ensemble Regression** trained on historical data  
- End-to-end **scikit-learn pipeline**  
- Model persistence using `joblib`

All predictions use the same pipeline that was trained on the dataset, ensuring consistency between training and inference.

---

## 🌐 Web Application Features

- Built using **Plotly Dash** and **Bootstrap**
- Manual vehicle detail input
- URL-based auto-fill using web scraping
- Scrapes data from multiple car listing websites
- Intelligent parsing of JSON-LD, metadata, and page text
- Price prediction in **EUR**
- Automatic conversion to **INR** using live exchange rates
- Interactive price gauge visualization
- **5-year depreciation forecast**

---

## 📁 Project Folder Structure

```text
Car-Price-Predictor/
│
├── assets/
│   ├── CarPricePredictor.png
│   └── style.css
│
├── data/
│   ├── CarPrices.csv
│   └── CleanedCarPrices.csv
│
├── joblib/
│   └── CarPricePipeline.joblib
│
├── app.py
├── CarPricePrediction.ipynb
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack
Python
Pandas
NumPy
Scikit-learn
Plotly Dash
BeautifulSoup (Web Scraping)
Joblib

---

## 🚀 How to Run the Application
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Run the app
```bash
python app.py
```
3. Open in browser
http://127.0.0.1:8050/

---

## 📈 Use Cases
Used car price estimation
Cross-platform vehicle valuation
Market trend analysis
End-to-end machine learning deployment portfolio project

---

## 🔮 Future Improvements
Model explainability using SHAP or LIME
Support for more vehicle categories
Cloud deployment (Docker / AWS / GCP)
Larger and region-specific datasets

## 👨‍💻 Author
### Abhinav Verma
Developed as a real-world Machine Learning project demonstrating dataset-driven modeling, web scraping, and interactive deployment using Plotly Dash.
