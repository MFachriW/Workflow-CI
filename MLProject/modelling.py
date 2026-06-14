import os
import sys
import io

# Fix Windows console encoding issues for emojis printed by MLflow
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

def train_model():
    # Preprocessed dataset path
    data_dir = 'house_prices_preprocessing'
    train_path = os.path.join(data_dir, 'train_preprocessed.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run preprocessing first!")
        return
        
    df = pd.read_csv(train_path)
    
    # Separate features and target
    X = df.drop(columns=['Id', 'SalePrice'])
    y = df['SalePrice']
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure MLflow (use local mlruns folder in GitHub CI, use localhost:5000 on student machine)
    if os.environ.get("CI") != "true":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
    # Only set experiment if not running inside an MLflow Project run
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("House_Price_Prediction")
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    print("Starting MLflow training run...")
    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Validation Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Log extra metrics manually (as backup/verification)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)
        
        # Save model locally for serving
        joblib.dump(model, 'model.joblib')
        print("Model saved to model.joblib successfully!")
        
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print("Training run completed and logged to MLflow UI!")

if __name__ == '__main__':
    train_model()
