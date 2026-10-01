import pandas as pd
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation-predictions-template.csv")


# Basic information
print("Training shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate load IDs:", df["load_id"].duplicated().sum())


# Target
print("\nPosted rate:")
print(df["posted_rate"].describe())

print("\nTarget quantiles:")
print(
    df["posted_rate"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)


# Date information
df["date"] = pd.to_datetime(df["date"])
validation["date"] = pd.to_datetime(validation["date"])

print("\nTraining dates:")
print(df["date"].min(), "to", df["date"].max())

print("\nValidation dates:")
print(validation["date"].min(), "to", validation["date"].max())


# Some useful distributions
print("\nEquipment:")
print(df["equipment"].value_counts())

print("\nLoads per month:")
print(df["date"].dt.to_period("M").value_counts().sort_index())


# Validation data
print("\nValidation shape:", validation.shape)

print("\nValidation missing values:")
print(validation.isnull().sum())

print("\nValidation equipment:")
print(validation["equipment"].value_counts())


# Submission template
print("\nTemplate shape:", template.shape)
print("Template columns:", template.columns.tolist())
print("IDs unique:", template["load_id"].nunique())