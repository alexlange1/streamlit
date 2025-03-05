# Unicorn Companies Analysis Web Application

## Overview
This web application provides an interactive analysis of unicorn companies, offering insights into their valuations, industries, geographic distribution, and investors. It also features a valuation predictor that estimates a startup's potential valuation based on key financial and industry factors.

## Accessing the Application
The web application is hosted on an AWS EC2 instance and can be accessed via:
- **Website:** [capitaltown.com](http://capitaltown.com)
- **Direct IP:** [54.161.199.20:8501](http://54.161.199.20:8501)

## Motivation
Initially, the application was built using a single dataset; however, it became evident that the dataset lacked critical features necessary for reliable valuation predictions. To improve accuracy, an additional dataset was incorporated, containing more comprehensive financial metrics such as **revenues and profitability**. These features significantly enhance the reliability of the valuation predictor.

## Features
- **Data Exploration:** Filter and analyze unicorn company data by industry, country, and valuation.
- **Interactive Visualizations:** Includes bar charts, box plots, and geographic maps using Plotly.
- **Startup News:** Fetches and displays the latest news related to startup funding and unicorns.
- **Investor Insights:** Highlights top venture capital firms investing in unicorns.
- **Trivia Game:** Test your knowledge about unicorn companies with an interactive quiz.
- **Startup Valuation Predictor:** Predicts a startup's valuation using a machine learning model based on RandomForestRegressor.

## Technology Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python, Pandas, NumPy, Scikit-Learn
- **Visualization:** Plotly, Plotly Express
- **Deployment:** AWS EC2, Streamlit

## Installation
To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unicorn-analysis.git
   cd unicorn-analysis
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser and navigate to `http://localhost:8501/`.

## Dataset Information
The application uses two datasets:
1. **Original Unicorn Companies Dataset:** Contains basic information about unicorn companies.
2. **Enhanced Dataset:** Adds crucial financial metrics such as revenue, profitability, and funding history to improve valuation prediction accuracy.

## Prediction Model
The valuation predictor uses a **RandomForestRegressor** trained on the enhanced dataset. Key factors include:
- Industry
- Country
- Annual Revenue
- Profitability
- Funding Stage

The model combines machine learning predictions with a **revenue-based valuation approach**, ensuring a more realistic estimate.

## Future Improvements
- **Expand Data Sources:** Incorporate more datasets for better accuracy.
- **Real-time Data Updates:** Fetch live unicorn company data.
- **Advanced ML Models:** Experiment with deep learning models for valuation predictions.

## License
This project is licensed under the MIT License.

## Author
Developed by [Alexander Lange](https://github.com/alexlange1)

