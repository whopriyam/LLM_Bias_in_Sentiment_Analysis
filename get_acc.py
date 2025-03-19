import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Load the dataset
df = pd.read_csv("data/output_with_sentiment_balanced_with_gpt.csv")

# Define the columns to compare
models = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "gpt-3.5-turbo",
    "gpt-4o"
]

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    return accuracy, f1score, precision, recall, fpr, fnr

# Compute metrics for each model
for model in models:
    print("*" * 75)
    print("\nMODEL NAME:", model)

    # Overall metrics
    accuracy, f1score, precision, recall, fpr, fnr = compute_metrics(df["label"], df[model])
    print(f"Accuracy for all SA: {accuracy}")
    print(f"F1 Score for all SA: {f1score}")
    print(f"Precision for all SA: {precision}")
    print(f"Recall for all SA: {recall}")
    print(f"False Positive Rate (FPR) for all SA: {fpr}")
    print(f"False Negative Rate (FNR) for all SA: {fnr}\n")

    # SAE (Standard American English) subset
    df_sae = df[df["sa"] == 1]
    accuracy, f1score, precision, recall, fpr, fnr = compute_metrics(df_sae["label"], df_sae[model])
    print(f"Accuracy for SAE: {accuracy}")
    print(f"F1 Score for SAE: {f1score}")
    print(f"Precision for SAE: {precision}")
    print(f"Recall for SAE: {recall}")
    print(f"False Positive Rate (FPR) for SAE: {fpr}")
    print(f"False Negative Rate (FNR) for SAE: {fnr}\n")

    # AAVE (African American Vernacular English) subset
    df_aave = df[df["sa"] == 0]
    accuracy, f1score, precision, recall, fpr, fnr = compute_metrics(df_aave["label"], df_aave[model])
    print(f"Accuracy for AAVE: {accuracy}")
    print(f"F1 Score for AAVE: {f1score}")
    print(f"Precision for AAVE: {precision}")
    print(f"Recall for AAVE: {recall}")
    print(f"False Positive Rate (FPR) for AAVE: {fpr}")
    print(f"False Negative Rate (FNR) for AAVE: {fnr}\n")
    print("*" * 75)
