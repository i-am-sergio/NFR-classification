# -*- coding: utf-8 -*-
"""
kfold_catboost.py (Clasificación de 10 Clases)

Este script evalúa un clasificador CatBoost en una tarea de clasificación de
10 clases, excluyendo las clases 'F' (Funcional) y 'PO' (Portabilidad),
utilizando una validación cruzada estratificada de 10 pliegues.

Pasos:
1.  Carga el dataset de embeddings desde el directorio `../../embeddings/`.
2.  Filtra el dataset para eliminar las clases 'F' y 'PO'.
3.  Codifica las etiquetas de texto a formato numérico.
4.  Configura un bucle de StratifiedKFold de 10 pliegues.
5.  DENTRO DEL BUCLE (para cada pliegue):
    a. Entrena un nuevo modelo CatBoost en 9 pliegues.
    b. Evalúa en el pliegue restante.
    c. Muestra un reporte de rendimiento detallado para el pliegue actual.
6.  FUERA DEL BUCLE:
    a. Calcula y muestra métricas promedio y la matriz de confusión agregada.
"""

print("--- Iniciando Pipeline: 10-Fold CV con CatBoost (Clasificación de 10 Clases) ---")

# ----------------------------------------
# 1. Importación de Librerías
# ----------------------------------------
import pandas as pd
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ----------------------------------------
# 2. Cargar y Procesar Datos
# ----------------------------------------
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    EMBEDDINGS_PATH = SCRIPT_DIR / ".." / ".." / "embeddings" / "embeddings.csv"
    df = pd.read_csv(EMBEDDINGS_PATH)
    print(f"Dataset de embeddings cargado desde '{EMBEDDINGS_PATH}'. Dimensiones: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: No se pudo encontrar el archivo 'embeddings.csv' en la ruta esperada.")
    sys.exit(1)

# --- Filtrado para 10 Clases (excluyendo 'F' y 'PO') ---
classes_to_remove = ['F', 'PO']
print(f"\nFiltrando el dataset para eliminar las clases: {classes_to_remove}...")
original_rows = len(df)
df = df[~df['class'].isin(classes_to_remove)].copy()
print(f"Filtrado completo. Se mantienen {len(df)} de {original_rows} filas.")
print("Distribución de las clases restantes:")
print(df['class'].value_counts())
# -------------------------------------------------------

X = df.drop('class', axis=1).values
y_text = df['class'].values
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y_text)
class_names = label_encoder.classes_
num_classes = len(class_names)
class_labels_ordered = np.arange(num_classes)
print(f"\nEtiquetas codificadas. Clases: {class_names}")

# ----------------------------------------
# 3. Configurar y Ejecutar el Bucle de K-Fold
# ----------------------------------------
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_accuracies, all_balanced_accuracies, all_classification_reports = [], [], []
all_w_precisions, all_w_recalls, all_w_f1s = [], [], []
total_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
task_type = 'GPU' if torch.cuda.is_available() else 'CPU'
print(f"CatBoost se ejecutará en: {task_type}")

for fold, (train_index, test_index) in enumerate(skf.split(X, y_numeric)):
    print(f"\n{'='*20} INICIANDO PLIEGUE {fold + 1}/{N_SPLITS} {'='*20}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_numeric[train_index], y_numeric[test_index]

    # Para clasificación multiclase, la loss_function es 'MultiClass'
    model = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, loss_function='MultiClass', task_type=task_type, random_state=42, verbose=0)
    
    print(f"Entrenando en {len(X_train)} muestras, validando en {len(X_test)} muestras...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print(f"\n--- Resultados del Pliegue {fold + 1} ---")
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nReporte de Clasificación del Pliegue:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    all_accuracies.append(acc)
    all_balanced_accuracies.append(balanced_acc)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    all_classification_reports.append(report_dict)
    all_w_precisions.append(report_dict['weighted avg']['precision'])
    all_w_recalls.append(report_dict['weighted avg']['recall'])
    all_w_f1s.append(report_dict['weighted avg']['f1-score'])
    total_conf_matrix += confusion_matrix(y_test, y_pred, labels=class_labels_ordered)

    print(f"Pliegue {fold + 1} completado.")

# ----------------------------------------
# 4. Mostrar Resultados Agregados
# ----------------------------------------
print(f"\n{'='*30} RESULTADOS FINALES AGREGADOS (CATBOOST - 10 CLASES) {'='*30}")
print("\n--- Métricas Globales (Promedio de 10 Pliegues) ---")
print(f"Accuracy:            {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
print(f"Balanced Accuracy:   {np.mean(all_balanced_accuracies):.4f} (+/- {np.std(all_balanced_accuracies):.4f})")
print(f"Weighted Precision:  {np.mean(all_w_precisions):.4f} (+/- {np.std(all_w_precisions):.4f})")
print(f"Weighted Recall:     {np.mean(all_w_recalls):.4f} (+/- {np.std(all_w_recalls):.4f})")
print(f"Weighted F1-Score:   {np.mean(all_w_f1s):.4f} (+/- {np.std(all_w_f1s):.4f})")
print("\n--- Métricas por Clase (Promedio de 10 Pliegues) ---")
for class_name in class_names:
    precisions = [report[class_name]['precision'] for report in all_classification_reports]
    recalls = [report[class_name]['recall'] for report in all_classification_reports]
    f1s = [report[class_name]['f1-score'] for report in all_classification_reports]
    
    print(f"Clase: {class_name}")
    print(f"  - Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"  - Recall:    {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"  - F1-Score:  {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print("\n--- Matriz de Confusión Agregada (Suma de 10 Pliegues) ---")
cm_df = pd.DataFrame(total_conf_matrix, index=class_names, columns=class_names)
print("Muestra el total de predicciones a lo largo de toda la validación cruzada:")
print(cm_df)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión Agregada (CatBoost - 10 Clases, 10-Fold CV)')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()
print("\n--- Fin del Pipeline ---")
