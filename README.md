# Data Analysis and Model Selection Application

This repository contains a Python application for data analysis and model selection. It utilizes Streamlit for creating an interactive web-based interface.

**Key Features:**

* **Data Loading and Processing:**
    - Supports loading data from CSV files or a SQL database.
    - Handles data cleaning and preprocessing.
    - Splits data into training and testing sets.
* **Model Training:**
    - Implements both genetic and exhaustive search algorithms for hyperparameter tuning.
    - Supports multiple machine learning algorithms such as LinearRegression, DecisionTreeRegressor, RandomForestRegressor, Lasso, Ridge and others.
    - Evaluates model performance using metrics like RMSE.
* **Interactive Visualization:**
    - Provides an exploratory data analysis (EDA) section with:
        - Data summaries (data types, missing values, descriptive statistics).
        - Histograms and boxplots for data visualization.
        - Correlation heatmaps.
    - Displays model results, including:
        - Comparison of RMSE between genetic and exhaustive search.
        - Best parameters found by each search method.
        - Table summarizing results for all algorithms and datasets.

**Getting Started:**

1. **Clone the repository:**
 ```bash
 git clone https://github.com/KainnT/Parameters-Genetic-Search.git
 ```
2. **Install dependencies:**

```bash
pip install -r requirements.txt 
(Note: Create a requirements.txt file listing all necessary libraries.)
```
3. **Configure data sources:**

If using SQL:
Update the db_params dictionary in load_and_process_data with your database credentials.
If using CSV files:
Update the csv_files list in load_and_process_data with the paths to your CSV files.

4. *Run the application:*
 ```bash
streamlit run app.py
 ```

*Usage:*

Navigate through the "EDA" and "Resultados" sections using the sidebar menu.
Interact with the application by selecting datasets, algorithms, and viewing the results.


*Contributing:*

Contributions are welcome! Please follow these guidelines:

Fork the repository.
Create a new branch for your feature or bug fix.   
Make your changes and commit them with clear and concise messages.
Push your changes to your fork.
Create a pull request.


---

## **License**
This project is licensed under the [MIT License](LICENSE).

---
