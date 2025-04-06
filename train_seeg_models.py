import glob
import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

dir = r"D:\Shaked_data\test1"
scores_csv_path = f"{dir}/scores.csv"
metadata = ['subj', 'epoch_id', 'chan_name', 'epoch']


def balance_dataset(X, y):
    num_ones = sum(y)
    min_count = min(num_ones, len(y) - num_ones)

    ones_idx = np.where(y == 1)[0]
    zeros_idx = np.where(y == 0)[0]
    selected_ones = np.random.choice(ones_idx, min_count, replace=False)
    selected_zeros = np.random.choice(zeros_idx, min_count, replace=False)

    selected_idx = np.concatenate([selected_ones, selected_zeros])
    X_balanced = X.iloc[selected_idx]
    y_balanced = y[selected_idx]
    return X_balanced, y_balanced


def train_lgbm(X_balanced, y_balanced, sub, channels):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_balanced, y_balanced)):
        X_train, X_test = X_balanced.iloc[train_idx], X_balanced.iloc[test_idx]
        y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]
        
        model = LGBMClassifier(verbose=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        tn, fp = confusion_matrix(y_test, y_pred).ravel()[:2]
        scores = {
            'sub': f"{sub}_{channels}", 
            'fold': str(fold + 1),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': float(tn / (tn + fp)),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'pr_auc': float(average_precision_score(y_test, y_prob))
        }
        metrics.append(scores)
    return metrics


def save_metrics(metrics, csv_path, sub, channels):
    metrics_df = pd.DataFrame(metrics)
    mean_scores = metrics_df.mean(numeric_only=True)
    mean_scores['fold'] = 'mean'
    mean_scores['sub'] = f"{sub}_{channels}"
    metrics_df = pd.concat([metrics_df, mean_scores.to_frame().T], ignore_index=True)

    write_mode = "a" if os.path.exists(csv_path) else "w"
    header = not(os.path.exists(csv_path))
    metrics_df.to_csv(csv_path, mode=write_mode, header=header, index=False)


def main():
    for file_path in glob.glob(f'{dir}/*.pkl'):
        info = file_path.split(".")[0].split("_")[-3:]
        channels = "_".join(info[:2])
        sub = info[2]

        with open(fr"{dir}\scalp_features_{channels}_{sub}.pkl", "rb") as f:
            sub_features = pickle.load(f)
        y_depth = np.load(fr"{dir}\y_depth_{sub}.npy")

        # Step 1: Remove metadata columns
        metadata_cols = [col for col in sub_features.columns if any(col.endswith(x) for x in metadata)]
        X = sub_features.drop(columns=metadata_cols)

        # Step 2: Balance dataset
        X_balanced, y_balanced = balance_dataset(X, y_depth)

        # Step 3: Train LGBM with K-Fold
        metrics = train_lgbm(X_balanced, y_balanced, sub, channels)

        # Step 4: Save the scores
        save_metrics(metrics, scores_csv_path, sub, channels)
        print(f"Finished {sub}_{channels}!")


if __name__ == "__main__":
    main()
