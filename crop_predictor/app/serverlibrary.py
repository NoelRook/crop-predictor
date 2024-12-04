import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from typing import TypeAlias
from typing import Optional, Any    
from flask import Flask, url_for
from app import db
from app.models import TrainingData 
app = Flask(__name__)

# Path to the CSV file for initial data import
filepath = os.path.join(app.static_folder, "crop yield data sheet.csv")

def getData(): 
    """
    Fetch data from the database, clean it, and separate it into features and target DataFrames.
    Returns:
        df_features (DataFrame): Independent variables (features).
        df_target (DataFrame): Dependent variable (target).
    """
    query_result = db.session.query(TrainingData).all()
    df_features = pd.DataFrame([{
        'Temperature': record.temperature,
        'Nitrogen (N)': record.nitrogen,
        'Potassium (K)': record.potassium
    } for record in query_result])

    df_target = pd.DataFrame([{
        'Yield': record.yield_
        } for record in query_result])
    #Data cleaning features 
    df_features.dropna(inplace=True)
    df_target.dropna(inplace=True)
    df_features['Temperature'] = pd.to_numeric(df_features['Temperature'], downcast='float')
    return df_features, df_target   # return back the clean df_features and df_target

# put Python code to prepare your features and target
def normalize_z(array: np.ndarray, 
                columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using Z-normalization.
    Args:
        array (np.ndarray): Array of feature data.
        columns_means (Optional[np.ndarray]): Pre-computed means of columns.
        columns_stds (Optional[np.ndarray]): Pre-computed standard deviations of columns.
    Returns:
        tuple: Normalized array, means, and standard deviations.
    """
    if columns_means is None or columns_stds is None:
        columns_means = array.mean(axis=0)
        columns_stds = array.std(axis=0)
    
    # Apply z-normalization
    out = (array - columns_means) / columns_stds
    return out, columns_means, columns_stds


def get_features_targets(df: pd.DataFrame, feature_names: list, target_names: list):
    """
    Extract features and targets from a DataFrame based on specified column names.
    Args:
        df (DataFrame): Input DataFrame.
        feature_names (list): List of feature column names.
        target_names (list): List of target column names.
    Returns:
        tuple: Features and targets DataFrames.
    """
    features = df.loc[:,feature_names]
    targets = df.loc[:,target_names]
    return features, targets

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    """
    Add an intercept column to the feature matrix.
    Args:
        np_feature (np.ndarray): Feature matrix.
    Returns:
        np.ndarray: Feature matrix with intercept column.
    """
    m, _ = np_feature.shape
    x0 = np.ones((m,1))
    return np.concatenate((x0, np_feature), axis=1)

def predict_linreg(array_feature: np.ndarray, 
                   beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Make predictions using linear regression.
    Args:
        array_feature (np.ndarray): Feature array.
        beta (np.ndarray): Model coefficients.
        means (Optional[np.ndarray]): Means for normalization.
        stds (Optional[np.ndarray]): Standard deviations for normalization.
    Returns:
        np.ndarray: Predictions.
    """
    if means is None and stds is None: 
        np_feature, _ , _ = normalize_z(array_feature)
    if means is None: 
        print('mean None')
        np_feature, _ , stds = normalize_z(array_feature,columns_stds=stds)
    if stds is None:
        np_feature, means , _ = normalize_z(array_feature,columns_means=means)
    if stds is not None and means is not None:
        np_feature, means , stds = normalize_z(array_feature,columns_means=means, columns_stds=stds)

    # Prepare feature matrix (add intercept column)
    X = prepare_feature(np_feature)

    # Calculate predictions using the linear regression formula y = Xb
    pred = calc_linreg(X, beta)
    return pred

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Perform linear regression prediction: y = Xb.
    Args:
        X (np.ndarray): Feature matrix.
        beta (np.ndarray): Coefficients.
    Returns:
        np.ndarray: Predicted values.
    """
    return np.matmul(X, beta)

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    Args:
        df_feature (DataFrame): Features DataFrame.
        df_target (DataFrame): Targets DataFrame.
        random_state (Optional[int]): Seed for reproducibility.
        test_size (float): Fraction of data to use as test set.
    Returns:
        tuple: Training and testing sets for features and targets.
    """
    # Set random seed for reproducibility if provided
    np.random.seed(random_state)
    
    # Calculate the number of test samples based on test_size
    total_samples = len(df_feature)
    test_samples = int(total_samples * test_size)
    
    # Randomly select test indices
    test_indices = np.random.choice(df_feature.index, size=test_samples, replace=False)
    
    # Create the test DataFrames using the selected indices
    df_feature_test = df_feature.loc[test_indices]
    df_target_test = df_target.loc[test_indices]
    
    # Create the train DataFrames by dropping the test indices
    df_feature_train = df_feature.drop(test_indices)
    df_target_train = df_target.drop(test_indices)
    
    # Return the four DataFrames as a tuple
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def compute_cost_linreg(X: np.ndarray, 
                        y: np.ndarray, 
                        beta: np.ndarray) -> np.ndarray:
    """
    Compute the cost function for linear regression.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        beta (np.ndarray): Coefficients.
    Returns:
        np.ndarray: Computed cost.
    """
    m, _ = X.shape
    # print(m)
    y_hat = calc_linreg(X,beta)
    delta = y_hat - y #this is a matrix subtraction
    #sq sum of delta
    delta_sq = delta ** 2 #this is a broacast function
    J = np.sum(delta_sq) / (2*m)
    return np.squeeze(J)

def gradient_descent_linreg(X: np.ndarray, 
                            y: np.ndarray, 
                            beta: np.ndarray, 
                            alpha: float, 
                            num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform gradient descent for linear regression.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target values.
        beta (np.ndarray): Initial coefficients.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations.
    Returns:
        tuple: Optimized coefficients and cost history.
    """
    
    m = len(y)  # number of training examples
    J_storage = np.zeros(num_iters)
    # To store the cost at each iteration
    # print(J_storage)
    for i in range(num_iters):
        # Calculate predictions
        y_hat = calc_linreg(X, beta)
        
        # Compute the gradient
        gradient = (1 / m) * np.matmul(X.transpose(), (y_hat - y))
        
        # Update beta
        beta = beta - alpha * gradient
        
        # Compute the cost for the current beta
        J_storage[i] = compute_cost_linreg(X, y, beta)
    
    return beta, J_storage

def trainModel():
    """
    Train a linear regression model using the data from the SQL database.
    Returns:
        tuple: Trained coefficients (beta), means, and standard deviations.
    """
    df_features, df_target = getData()
    # 2. Splitting of our df_feautres and df_target into training and testing datasets.
    df_features_train, df_features_test, df_target_train, _ = split_data(df_features, df_target, 100, 0.3)

    # 3. Normalisation of the training and testing datasets. 
    df_features_train,means,stds = normalize_z(df_features_train.to_numpy()) 
    df_features_test, _, _ = normalize_z(df_features_test)

    # 4. Calling of linear regression prediction and gradient descent functions with normalised data sets as inputs. 
    X: np.ndarray = prepare_feature(df_features_train)
    target: np.ndarray = df_target_train.to_numpy()

    iterations: int = 1500  
    alpha: float = 0.01 #learning rate
    beta: np.ndarray = np.zeros((4,1))

    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
    pred: np.ndarray = predict_linreg(df_features_test.to_numpy(), beta, means, stds)
    return beta, means, stds
