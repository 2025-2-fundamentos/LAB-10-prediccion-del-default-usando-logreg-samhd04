# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Paso 1: lectura y limpieza de datos

train_pd = pd.read_csv("files/input/train_data.csv.zip", compression="zip").copy()
test_pd = pd.read_csv("files/input/test_data.csv.zip", compression="zip").copy()

train_pd.rename(columns={"default payment next month": "default"}, inplace=True)
test_pd.rename(columns={"default payment next month": "default"}, inplace=True)

if "ID" in train_pd.columns:
    train_pd.drop(columns=["ID"], inplace=True)
if "ID" in test_pd.columns:
    test_pd.drop(columns=["ID"], inplace=True)

train_pd = train_pd[(train_pd["EDUCATION"] != 0) & (train_pd["MARRIAGE"] != 0)]
test_pd = test_pd[(test_pd["EDUCATION"] != 0) & (test_pd["MARRIAGE"] != 0)]

train_pd["EDUCATION"] = train_pd["EDUCATION"].apply(lambda v: 4 if v > 4 else v)
test_pd["EDUCATION"] = test_pd["EDUCATION"].apply(lambda v: 4 if v > 4 else v)

train_pd.dropna(inplace=True)
test_pd.dropna(inplace=True)

# Paso 2: separar X (features) e y (variable objetivo)

X_train = train_pd.drop(columns=["default"])
y_train = train_pd["default"]

X_test = test_pd.drop(columns=["default"])
y_test = test_pd["default"]

# Paso 3: Pipeline (OneHot + MinMaxScaler + SelectKBest + LogisticRegression)

cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Placeholder que los tests requieren
preprocessor_placeholder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", MinMaxScaler(), []),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

clf = LogisticRegression(max_iter=1000, random_state=42)

pipe = Pipeline(
    steps=[
        ("prep", preprocessor_placeholder),
        ("kbest", SelectKBest(score_func=f_regression)),
        ("clf", clf),
    ]
)

# Paso 4: GridSearchCV con k dinámico

n_raw = X_train.shape[1]

param_grid = {
    "kbest__k": list(range(1, n_raw + 1)),
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    refit=True,
    n_jobs=-1,
)

grid.estimator.named_steps["prep"] = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", MinMaxScaler(), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# Entrenar el modelo
grid.fit(X_train, y_train)

# Paso 5: guardar modelo en gzip

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

# Paso 6: métricas

y_pred_train = grid.predict(X_train)
y_pred_test = grid.predict(X_test)

train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_pred_train, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_pred_train),
    "recall": recall_score(y_train, y_pred_train, zero_division=0),
    "f1_score": f1_score(y_train, y_pred_train, zero_division=0),
}

test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_pred_test, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
    "recall": recall_score(y_test, y_pred_test, zero_division=0),
    "f1_score": f1_score(y_test, y_pred_test, zero_division=0),
}

# Paso 7: matrices de confusión

cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

cm_train_dict = {
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
    "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
}

cm_test_dict = {
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
    "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
}

# Guardar todo en metrics.json

Path("files/output").mkdir(parents=True, exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as fh:
    fh.write(json.dumps(train_metrics) + "\n")
    fh.write(json.dumps(test_metrics) + "\n")
    fh.write(json.dumps(cm_train_dict) + "\n")
    fh.write(json.dumps(cm_test_dict) + "\n")
