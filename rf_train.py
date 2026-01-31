
import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,precision_score,recall_score,f1_score


# Task-1: Data loading

df = pd.read_csv("vgsales.csv")

print(df.head(10))


# Task-2: Data preprocessing

X = df.drop(['Global_Sales','Rank'], axis=1)
y = df["Global_Sales"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

#Task-3: Pipeline Creation (preprocessor + model)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

#Task-5: Training Model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

#Task-6: Cross Validation

cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring="r2"
)

print("Mean CV Score:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

#Task-7: Hyperparameter Tuning

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

#Task-8: Best Model Selection

best_model = grid.best_estimator_

#Task-9: Model Performance Evaluation

y_pred = best_model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#
with open("VG_rf_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline,f)

    print("Random Forest pipeline saved as VG_rf_pipeline.pkl")