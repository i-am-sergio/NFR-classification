# -*- coding: utf-8 -*-
"""
kfold_lr.py (Clasificación de 10 Clases con SMOTExT en cada pliegue)

CORRECCIÓN: Se soluciona el error 'ValueError: cannot reindex...' eliminando
la columna 'class' duplicada del DataFrame de embeddings ANTES de la
concatenación. Esto asegura que el DataFrame final tenga nombres de columna únicos.
"""

print("--- Iniciando Pipeline: 10-Fold CV con LR y SMOTExT en cada Pliegue ---")

# ----------------------------------------
# 1. Importación de Librerías
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# ----------------------------------------
# 2. Cargar y Procesar Datos
# ----------------------------------------
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    EMBEDDINGS_PATH = SCRIPT_DIR / ".." / ".." / "embeddings" / "embeddings.csv"
    df_embeddings = pd.read_csv(EMBEDDINGS_PATH)
    
    DATASET_PATH = SCRIPT_DIR / ".." / ".." / ".." /"dataset" / "datos_limpios.csv" 
    df_text = pd.read_csv(DATASET_PATH)
    
    # ----- SOLUCIÓN AL ERROR: Eliminar la columna 'class' duplicada -----
    # El DataFrame de embeddings ya contiene la clase, la eliminamos para evitar duplicados.
    if 'class' in df_embeddings.columns:
        df_embeddings = df_embeddings.drop('class', axis=1)
    # ---------------------------------------------------------------------

    # Fusionamos para tener texto y embeddings juntos
    df = pd.concat([df_text, df_embeddings], axis=1)
    
    # Resetear el índice sigue siendo una buena práctica para evitar otros problemas
    df = df.reset_index(drop=True)

    print(f"Datasets cargados, fusionados y con columnas únicas. Dimensiones: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: No se pudieron encontrar los archivos 'embeddings.csv' o 'datos_limpios.csv'.")
    sys.exit(1)

# Ahora el filtrado funcionará sin problemas
classes_to_remove = ['F', 'PO']
print(f"\nFiltrando el dataset para eliminar las clases: {classes_to_remove}...")
df = df[~df['class'].isin(classes_to_remove)].copy()
print(f"Filtrado completo. Se mantienen {len(df)} filas.")

X_embeddings = df.filter(regex='^embedding_').values
X_text = df['RequirementText'].values
y_text = df['class'].values
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y_text)
class_names = label_encoder.classes_
num_classes = len(class_names)
class_labels_ordered = np.arange(num_classes)
print(f"\nEtiquetas codificadas. Clases: {class_names}")

# ----------------------------------------
# 3. Configurar Modelos y Bucle de K-Fold
# (El resto del script no necesita cambios y debería funcionar ahora)
# ----------------------------------------
# ... (el resto del código es idéntico al que te pasé antes) ...
encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
decoder_pipeline = pipeline('text-generation', model='distilgpt2', device=0 if torch.cuda.is_available() else -1)

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_accuracies, all_balanced_accuracies, all_classification_reports = [], [], []
total_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for fold, (train_index, test_index) in enumerate(skf.split(X_embeddings, y_numeric)):
    print(f"\n{'='*20} INICIANDO PLIEGUE {fold + 1}/{N_SPLITS} {'='*20}")

    X_train_emb, X_test_emb = X_embeddings[train_index], X_embeddings[test_index]
    X_train_text, _ = X_text[train_index], X_text[test_index]
    y_train, y_test = y_numeric[train_index], y_numeric[test_index]
    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled_emb, y_train_resampled = smote.fit_resample(X_train_emb, y_train)
    
    num_original_samples = len(X_train_emb)
    synthetic_embeddings = X_train_resampled_emb[num_original_samples:]
    
    if len(synthetic_embeddings) > 0:
        print(f"Se generaron {len(synthetic_embeddings)} embeddings sintéticos. Decodificando a texto...")
        similarity_scores = util.cos_sim(synthetic_embeddings, X_train_emb)
        closest_indices = torch.argmax(similarity_scores, dim=1)
        closest_texts = [X_train_text[idx] for idx in closest_indices]
        
        prompts = [f"Paraphrase this requirement: '{text}'" for text in closest_texts]
        generated_texts = decoder_pipeline(prompts, max_new_tokens=30, num_return_sequences=1, pad_token_id=decoder_pipeline.tokenizer.eos_token_id)
        synthetic_texts = [gen[0]['generated_text'].replace(prompt, "").strip() for gen, prompt in zip(generated_texts, prompts)]
        
        print("Re-codificando texto sintético a embeddings...")
        new_embeddings = encoder_model.encode(synthetic_texts, show_progress_bar=False)
        
        X_train_resampled_emb[num_original_samples:] = new_embeddings
        print("SMOTExT completado.")
    else:
        print("No se necesitaron muestras sintéticas en este pliegue.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled_emb)
    X_test_scaled = scaler.transform(X_test_emb)
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    
    print(f"Entrenando en {len(X_train_scaled)} muestras (aumentadas), validando en {len(X_test_emb)} muestras...")
    model.fit(X_train_scaled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)
    
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
    total_conf_matrix += confusion_matrix(y_test, y_pred, labels=class_labels_ordered)

    print(f"Pliegue {fold + 1} completado.")
    
# ... (el resto del código de reporte) ...
# ----------------------------------------
# 4. Mostrar Resultados Agregados
# ----------------------------------------
print(f"\n{'='*30} RESULTADOS FINALES AGREGADOS (LR + SMOTETomek - 10 CLASES) {'='*30}")
print("\n--- Métricas Globales (Promedio de 10 Pliegues) ---")
print(f"Accuracy:            {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
print(f"Balanced Accuracy:   {np.mean(all_balanced_accuracies):.4f} (+/- {np.std(all_balanced_accuracies):.4f})")
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
plt.title('Matriz de Confusión Agregada (LR + SMOTETomek - 10 Clases, 10-Fold CV)')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()
print("\n--- Fin del Pipeline ---")
