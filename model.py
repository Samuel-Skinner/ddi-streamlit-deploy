# Main
import pandas as pd

# Split and Cross Val
from sklearn.model_selection import train_test_split

# Preprocess and Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression

# Ensemble Models
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import (mean_absolute_error, 
                             mean_absolute_error)

# Environments and serialization
import joblib

# formatting
import datetime

# Datasets
from sklearn.datasets import load_diabetes

def model_saving(features, target, model_pipeline, mae, feature_importance, name='auto_model.joblib'):
    """Note: the model used must be the last step in the model_pipeline"""
    
    metadata = {
    'model': model_pipeline,
    'date/time': datetime.datetime.now(),
    'performance': f'Mean Absolute Error is {mae}',
    'model used': model_pipeline.steps[-1][1],
    'features': features,
    'target': target,
    'mean absolute error': mae,
    'feature importance': feature_importance
    }

    joblib.dump(metadata, name)

cars_df = pd.read_csv("data/cars.csv")
cars_df["horsepower"] = pd.to_numeric(cars_df["horsepower"], errors="coerce")
cars_df = cars_df.dropna()
X = cars_df.drop(["mpg", "car name", "model year"], axis=1)
y = cars_df["mpg"]
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42)

numeric_features = ["horsepower", "displacement", "weight", "acceleration"]
categorical_features = ["cylinders", "origin"]

numeric_transformer = Pipeline(steps=[
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


forest = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor())
])

forest.fit(X_train, y_train)
y_preds = forest.predict(X_test)

mae_for = mean_absolute_error(y_test, y_preds)
importance = pd.Series(forest["model"].feature_importances_, index=forest["preprocess"].get_feature_names_out())

model_saving(X.columns, y.name, forest, mae_for, importance, 'data/cars_mpg_forest.joblib')

model_pipeline = Pipeline(steps=[

    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

model_pipeline.fit(X_train, y_train)
y_preds = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_preds)

model_saving(X.columns, y.name, model_pipeline, mae, importance, 'data/cars_mpg_predictor.joblib')