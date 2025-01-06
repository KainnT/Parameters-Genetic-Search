import plotly.graph_objs as go
import plotly.express as px
from sklearn.exceptions import ConvergenceWarning
import warnings
import threading
import seaborn as sns
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import database_exists, create_database
import pymssql
import time
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import CGenerator as cG
import CEvaluator as cE
import CPredictor as cP
import CVisualizer as cV
import streamlit as st



def main():
    # Sql connection
    db_params = {
        "username": " ",
        "password": " ",
        "host": " ",
        "database": " ",
        "port": 000
    }
    # Csv path
    start_time = time.time()
    use_sql = True
    file_paths = "Datasets"
    csv_files = []

    for filename in os.listdir(file_paths):
        if filename.endswith(".csv"):
            full_path = os.path.join(file_paths, filename)
            print(f"Reading file: {full_path}")
            csv_files.append(full_path)
            
    if not csv_files:
        st.error("No dataset encontrado")
        return


    X_trains, X_tests, y_trains, y_tests, dfs = cV.load_and_process_data(csv_files, use_sql, db_params)

    results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation = cV.train_models(X_trains, X_tests, y_trains, y_tests)

    visualizer = cV.Visualizer(genetic_evaluation, exhaustive_evaluation, dfs)
    visualizer.create_dashboard()


if __name__ == "__main__":
    main()