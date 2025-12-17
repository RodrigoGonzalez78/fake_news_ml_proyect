import yaml
import os
import sys
import pandas as pd
import time  # <--- NUEVO: Para medir tiempo
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Agregar ruta base para imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.build_features import load_and_process_data
from model_arch import build_model_architecture

def run_training():
    # 1. Cargar Configuración
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['paths']['output_models'], exist_ok=True)

    # 2. Procesar Datos (Una vez para todos)
    print("--- PREPARANDO DATOS ---")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_process_data(config)

    results = []

    # 3. Ejecutar los 3 Experimentos
    for exp in config['experiments']:
        print(f"\n{'='*50}")
        print(f" ENTRENANDO: {exp['name']}")
        print(f"{'='*50}")
        
        # Construir Modelo
        model = build_model_architecture(
            config['global_params']['vocab_size'],
            config['global_params']['max_length'],
            exp
        )
        
        # Verificar parámetros
        params_count = model.count_params()
        print(f"-> Arquitectura: {exp['type'].upper()}")
        print(f"-> Unidades: {exp['units']}")
        print(f"-> Total Parámetros: {params_count:,}")
        
        # Configurar Early Stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        # --- INICIO CRONÓMETRO ---
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            epochs=config['global_params']['epochs'],
            batch_size=config['global_params']['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # --- FIN CRONÓMETRO ---
        end_time = time.time()
        training_time = end_time - start_time
        print(f"-> Tiempo de entrenamiento: {training_time:.2f} segundos")

        # --- EVALUACIÓN EXHAUSTIVA ---
        # Hacemos predicciones sobre el Test Set
        y_pred_prob = model.predict(X_test, verbose=0)
        # Convertimos probabilidades a 0 o 1 (usando 0.5 como umbral)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calcular todas las métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        # Desglosar Matriz de Confusión (Vital para el informe)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"-> Resultados: Acc={acc:.2%} | F1={f1:.2%} | Time={training_time:.1f}s")
        
        # Guardar Modelo
        model.save(os.path.join(config['paths']['output_models'], f"{exp['name']}.keras"))
        
        # Guardar TODO en la lista de resultados
        results.append({
            "Experimento": exp['name'],
            "Tipo": exp['type'],
            "Unidades": exp['units'],
            "Parámetros": params_count,
            "Tiempo (seg)": round(training_time, 2),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),  # ¿Qué tan confiable es cuando dice FAKE?
            "Recall": round(rec, 4),      # ¿Cuántos FAKES atrapó del total?
            "F1-Score": round(f1, 4),     # Balance entre Precision y Recall
            "AUC-ROC": round(auc, 4),     # Capacidad de distinción general
            "Falsos Positivos": fp,       # Noticias reales marcadas como fake (Error grave)
            "Falsos Negativos": fn        # Fakes que se escaparon (Error grave)
        })

    # 4. Generar Reporte Completo
    print("\n\n--- REPORTE FINAL DETALLADO ---")
    df_res = pd.DataFrame(results)
    
    # Reordenar columnas para que sea más legible
    cols = ["Experimento", "Tiempo (seg)", "Accuracy", "F1-Score", "Recall", "Precision", "Parámetros", "Falsos Positivos", "Falsos Negativos"]
    print(df_res[cols])
    
    # Guardar CSV
    csv_path = os.path.join(config['paths']['output_models'], "resultados_finales_completo.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nReporte guardado en: {csv_path}")

if __name__ == "__main__":
    run_training()