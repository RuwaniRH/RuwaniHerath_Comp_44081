Public Service Commission-Written exam - #44081

1. The python code is given in the GitHub repository.
2. The python code is given in the GitHub repository.
3. **Client-Friendly Summary**

- I built two forecasting models: one for **redemptions** (ferry boardings) and one for **ticket sales**.
- These models help the City predict how many people will ride or buy tickets on a daily basis.
- The models are trained on historical ferry activity data and consider:
  - **Time patterns** (day of the week, month, holidays)
  - **Recent activity** (e.g., number of riders yesterday or last week)
- We used advanced tools like **Prophet**, which can account for trends, seasonal patterns, and uncertainty.
- For redemptions, we tested other models too, like **XGBoost** and **SARIMAX**, and found that Prophet with added memory (lags) worked best overall.
- We built a separate model for ticket sales using similar techniques, so planning can happen earlier in the customer journey.
- Each model provides not just a number, but a **range of possible values**, giving a clearer picture of best- and worst-case scenarios.
- The system is **modular**, easy to update, and can be automated.
- All results were tested thoroughly and validated using industry-standard techniques, making them reliable for planning and decision-making.

4. **Technical Summary**

I developed robust and interpretable models to forecast ferry **redemptions** and **ticket sales** using historical data from the Toronto Island Ferry service. These models support daily operational planning and long-term resource allocation.

**Data Ingestion and Feature Engineering**

I began by ingesting timestamped ticketing data and aggregating it to the daily level. The dataset was enriched with engineered features that help models learn patterns more effectively. These included

- **Lag features** (e.g., Redemption_t–1, Redemption_t–7, Sales_t–1)
- **Rolling averages** over 7 and 30 days to smooth short-term noise
- **Calendar-based features** (day of week, month, quarter, weekend flags, holiday indicators)
- **Cross-lag features**, e.g., previous sales as predictors for future redemptions

**Redemption Forecasting Models**

For redemptions, I tested three modeling approaches.

- **Prophet**: Primary model for redemptions. Captured seasonality, holidays, and uncertainty using trend decomposition and forecast intervals. Incorporating lag features improved performance significantly.
- **SARIMAX**: Used to model redemptions with exogenous regressors. Produced interpretable forecasts with 95% confidence intervals.
- **XGBoost**: Tree-based model trained on engineered features. We applied bootstrapped ensembles and quantile regression to provide uncertainty bands.

**Sales Forecasting Models**

A parallel process was used to model ticket sales.

- **Prophet**: Adapted to model ticket sales as count data, capturing trends and cycles.
- **Poisson GLM**: Considered as an alternative model for count data, integrating lag and seasonal components, with parametric confidence intervals.

**Evaluation**

For model evaluation, I used rolling-origin back testing to simulate how the models would perform in real-time deployment. I assessed forecast accuracy with Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Additionally, I evaluated the models’ uncertainty estimates by comparing actual data coverage within the predicted confidence intervals. Prophet models with engineered features demonstrated a ~15% improvement in RMSE over naïve baselines, and prediction intervals maintained 90% coverage at a 95% nominal level.

**Software Engineering and Deployment**

- The codebase was fully modular:
  - data_loader.py: Ingests and aggregates data
  - features.py: Adds temporal and statistical features
  - Improved_Forcasting_model.py and Sales_forecasting_model.py: Define model pipelines
  - Evaluate.py: Handles forecast validation and metrics
- Version control was maintained via Git with feature-based branching.
- All dependencies are listed in requirements.txt for reproducibility.
- Key data transformations and model outputs were tested using pytest.
- A unified pipeline script (main.py) can generate forecasts and plots in a single execution, supporting automation or integration into a dashboard.

This setup ensures that the forecasting system is transparent, accurate, and ready for operational use or future enhancement.
