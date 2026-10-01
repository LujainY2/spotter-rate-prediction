import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRAIN_PATH = "data/train-test.csv"
VALIDATION_PATH = "data/validation.csv"
TEMPLATE_PATH = "data/validation-predictions-template.csv"

df = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
template = pd.read_csv(TEMPLATE_PATH)


def add_features(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    data["route"] = (
        data["pickup"].astype(str) + "_" +
        data["delivery"].astype(str)
    )

    data["distance_per_weight"] = (
        data["distance"] / data["weight"].replace(0, np.nan)
    )

    return data


df = add_features(df)
validation = add_features(validation)


train_df = df[df["date"] < "2025-09-01"]
val_df = df[df["date"] >= "2025-09-01"]


TARGET = "posted_rate"

FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "route",
    "distance_per_weight",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]


model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
)


val_predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, val_predictions)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


X_full = df[FEATURES]
y_full = df[TARGET]
X_final = validation[FEATURES]


final_model = CatBoostRegressor(
    iterations=model.get_best_iteration(),
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)

final_model.fit(
    X_full,
    y_full,
    cat_features=CATEGORICAL_FEATURES,
)


final_predictions = final_model.predict(X_final)
final_predictions = np.maximum(final_predictions, 0.01)


predictions = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": final_predictions,
})

predictions = template[["load_id"]].merge(
    predictions,
    on="load_id",
    how="left",
)

if predictions["predicted_rate"].isna().any():
    raise ValueError("Some validation IDs did not receive predictions.")

predictions.to_csv("validation_predictions.csv", index=False)

print("Created validation_predictions.csv")