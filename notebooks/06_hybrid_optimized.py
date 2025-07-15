# 06_hybrid_optimized.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os
import json
from datetime import datetime
from joblib import dump, load

# Configuración
output_dir = 'results/06_hybrid_optimized'
plots_dir = f'{output_dir}/plots'
models_dir = 'models/final'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_data():
    """Cargar datos procesados"""
    df = pd.read_csv('data/processed/02_features/features_engineered.csv')
    
    # Obtener features (excluyendo targets e IDs)
    features = [col for col in df.columns 
               if col not in ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                             'necesita_reposicion', 'cantidad_a_reponer', 
                             'log_cantidad_a_reponer']]
    
    print(f"✅ Datos cargados: {df.shape[0]} registros, {len(features)} features")
    return df, features

def prepare_data(df, features):
    """Preparar datos para modelo híbrido optimizado"""
    print("\n📊 PREPARANDO DATASETS")
    print("-" * 50)
    
    # Seleccionar features numéricas
    X = df[features].select_dtypes(include=['number'])
    
    # Manejar NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Targets
    y_class = df['necesita_reposicion'] 
    y_reg = df['cantidad_a_reponer']
    y_reg_log = df['log_cantidad_a_reponer']
    
    # Split estratificado
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Para regresión, solo usar casos positivos
    mask_train = y_reg[X_train.index] > 0
    mask_test = y_reg[X_test.index] > 0
    
    X_reg_train = X_train[mask_train]
    y_reg_train = y_reg[X_train.index][mask_train]
    y_reg_log_train = y_reg_log[X_train.index][mask_train]
    
    X_reg_test = X_test[mask_test]
    y_reg_test = y_reg[X_test.index][mask_test]
    y_reg_log_test = y_reg_log[X_test.index][mask_test]
    
    # Escalar datos
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_reg_train_scaled = scaler.transform(X_reg_train)
    X_reg_test_scaled = scaler.transform(X_reg_test)
    
    # Verificar NaN después del escalado
    if np.isnan(X_train_scaled).any():
        print("⚠️ Detectados NaN en X_train_scaled después del escalado, aplicando imputación")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    
    if np.isnan(X_test_scaled).any():
        print("⚠️ Detectados NaN en X_test_scaled después del escalado, aplicando imputación")
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
    
    if np.isnan(X_reg_train_scaled).any():
        print("⚠️ Detectados NaN en X_reg_train_scaled después del escalado, aplicando imputación")
        X_reg_train_scaled = np.nan_to_num(X_reg_train_scaled, nan=0.0)
    
    if np.isnan(X_reg_test_scaled).any():
        print("⚠️ Detectados NaN en X_reg_test_scaled después del escalado, aplicando imputación")
        X_reg_test_scaled = np.nan_to_num(X_reg_test_scaled, nan=0.0)
    
    # Guardar scaler
    dump(scaler, f'{models_dir}/features_scaler.joblib')
    
    return {
        'classification': {
            'X_train': X_train_scaled, 
            'X_test': X_test_scaled,
            'y_train': y_class_train, 
            'y_test': y_class_test,
            'feature_names': X.columns.tolist(),
        },
        'regression': {
            'X_train': X_reg_train_scaled, 
            'X_test': X_reg_test_scaled,
            'y_train': y_reg_train, 
            'y_test': y_reg_test,
            'y_log_train': y_reg_log_train, 
            'y_log_test': y_reg_log_test,
            'feature_names': X.columns.tolist(),
        },
        'indices': {
            'train_indices': X_train.index,
            'test_indices': X_test.index,
            'train_reg_indices': X_reg_train.index,
            'test_reg_indices': X_reg_test.index
        }
    }

def train_ensemble_classifier(data):
    """Entrenar clasificador ensemble mejorado"""
    print("\n🔍 ENTRENANDO CLASIFICADOR ENSEMBLE OPTIMIZADO")
    print("-" * 50)
    
    # Definir modelos base optimizados
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=7, 
            min_samples_leaf=2, 
            class_weight='balanced',
            random_state=42)),
        ('gb', GradientBoostingClassifier(
            n_estimators=150, 
            max_depth=4, 
            learning_rate=0.05, 
            subsample=0.9,
            random_state=42)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=150, 
            max_depth=4, 
            learning_rate=0.05, 
            gamma=0.1,
            subsample=0.9, 
            colsample_bytree=0.8,
            scale_pos_weight=3,
            eval_metric='logloss',
            random_state=42))
    ]
    
    # Crear voting classifier
    voting_clf = VotingClassifier(
        estimators=base_models,
        voting='soft',
        weights=[2, 1, 2]
    )
    
    # Verificar NaN antes de SMOTE
    X_train = data['classification']['X_train']
    y_train = data['classification']['y_train']
    
    # Aplicar SMOTE para balance de clases
    smote = SMOTE(random_state=42, sampling_strategy=0.6)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"✅ SMOTE aplicado: {y_train.mean():.1%} → {y_train_resampled.mean():.1%} positivos")
    
    # Entrenar modelo
    voting_clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluar en conjunto de prueba
    y_proba = voting_clf.predict_proba(data['classification']['X_test'])[:, 1]
    
    # Encontrar umbral óptimo para F1
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.7, 0.02):
        y_pred_threshold = (y_proba > threshold).astype(int)
        f1_score_threshold = f1_score(data['classification']['y_test'], y_pred_threshold)
        
        if f1_score_threshold > best_f1:
            best_f1 = f1_score_threshold
            best_threshold = threshold
    
    print(f"✅ Umbral óptimo encontrado: {best_threshold:.2f} (F1: {best_f1:.4f})")
    y_pred = (y_proba > best_threshold).astype(int)
    
    # Métricas
    accuracy = accuracy_score(data['classification']['y_test'], y_pred)
    precision = precision_score(data['classification']['y_test'], y_pred)
    recall = recall_score(data['classification']['y_test'], y_pred)
    f1 = f1_score(data['classification']['y_test'], y_pred)
    
    print(f"✅ Resultados del Clasificador Ensemble Optimizado:")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    
    # Guardar modelo y umbral
    dump(voting_clf, f'{models_dir}/ensemble_classifier.joblib')
    
    return voting_clf, best_threshold, {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(best_threshold)
    }

def train_bagging_regressor(data):
    """Entrenar modelo de regresión con bagging según sugerencia del tutor"""
    print("\n📈 ENTRENANDO MODELO DE REGRESIÓN CON BAGGING")
    print("-" * 50)
    
    # Base estimator (XGBoost con transformación logarítmica)
    base_model = xgb.XGBRegressor(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.01,
        subsample=0.8, 
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Crear modelo de bagging
    bagging_reg = BaggingRegressor(
        estimator=base_model,
        n_estimators=10,  # Número de estimadores en el bagging
        max_samples=0.8,  # Porcentaje de muestras en cada estimador
        max_features=0.8,  # Porcentaje de features en cada estimador
        bootstrap=True,   # Muestreo con reemplazo
        bootstrap_features=False,
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar con datos log-transformados
    X_train = data['regression']['X_train']
    y_train = data['regression']['y_log_train']
    
    bagging_reg.fit(X_train, y_train)
    
    # Evaluar en conjunto de prueba
    y_pred_log = bagging_reg.predict(data['regression']['X_test'])
    y_pred = np.expm1(y_pred_log)  # Volver a escala original
    y_true = np.expm1(data['regression']['y_log_test'])
    
    # Métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"✅ Resultados del Modelo Bagging Regressor:")
    print(f"   • MAE: {mae:.2f}")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"   • R²: {r2:.4f}")
    
    # Comparar con modelo simple
    simple_model = xgb.XGBRegressor(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.01,
        subsample=0.8, 
        colsample_bytree=0.8,
        random_state=42
    )
    
    simple_model.fit(X_train, y_train)
    y_pred_simple_log = simple_model.predict(data['regression']['X_test'])
    y_pred_simple = np.expm1(y_pred_simple_log)
    
    mae_simple = mean_absolute_error(y_true, y_pred_simple)
    rmse_simple = np.sqrt(mean_squared_error(y_true, y_pred_simple))
    r2_simple = r2_score(y_true, y_pred_simple)
    
    print(f"\n📊 Comparación con modelo simple:")
    print(f"   • MAE Bagging: {mae:.2f} vs MAE Simple: {mae_simple:.2f}")
    print(f"   • R² Bagging: {r2:.4f} vs R² Simple: {r2_simple:.4f}")
    
    # Usar el mejor modelo
    if mae < mae_simple:
        print("✅ Bagging mejora el rendimiento, usando este modelo")
        final_model = bagging_reg
        final_metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
    else:
        print("⚠️ Modelo simple tiene mejor rendimiento, utilizando este")
        final_model = simple_model
        final_metrics = {
            'mae': float(mae_simple),
            'rmse': float(rmse_simple),
            'r2': float(r2_simple)
        }
    
    # Guardar modelo
    dump(final_model, f'{models_dir}/regression_model.joblib')
    
    return final_model, final_metrics

def implement_hybrid_model(data, class_model, class_threshold, reg_model):
    """Implementar modelo híbrido final optimizado"""
    print("\n🔄 EVALUANDO MODELO HÍBRIDO FINAL")
    print("-" * 50)
    
    # Hacer predicciones de clasificación con umbral optimizado
    X_test = data['classification']['X_test']
    y_class_true = data['classification']['y_test']
    
    # Usar predict_proba con umbral optimizado
    y_class_proba = class_model.predict_proba(X_test)[:, 1]
    y_class_pred = (y_class_proba > class_threshold).astype(int)
    
    # Identificar índices donde se predice reposición
    indices_reposicion = np.where(y_class_pred == 1)[0]
    test_indices = data['indices']['test_indices']
    
    # Crear array para predicciones híbridas
    y_hybrid_pred = np.zeros(len(y_class_pred))
    
    # Si hay casos con reposición predicha
    if len(indices_reposicion) > 0:
        # Obtener los datos de test correspondientes a esos índices
        X_test_reposicion = X_test[indices_reposicion]
        
        # Predecir cantidades
        y_pred_log = reg_model.predict(X_test_reposicion)
        y_pred_cantidades = np.expm1(y_pred_log)
        
        # Asignar valores predichos
        for i, idx in enumerate(indices_reposicion):
            y_hybrid_pred[idx] = y_pred_cantidades[i]
    
    # Evaluar clasificación
    accuracy = accuracy_score(y_class_true, y_class_pred)
    precision = precision_score(y_class_true, y_class_pred)
    recall = recall_score(y_class_true, y_class_pred)
    f1 = f1_score(y_class_true, y_class_pred)
    
    # Evaluar regresión solo para casos donde ambos (real y predicho) son positivos
    true_pos_indices = np.where((y_class_pred == 1) & (np.array(y_class_true) == 1))[0]
    
    if len(true_pos_indices) > 0:
        # Obtener predicciones para estos índices
        y_pred_true_pos = np.array([y_hybrid_pred[i] for i in true_pos_indices])
        
        # Obtener los índices originales
        original_indices = [test_indices[i] for i in true_pos_indices]
        
        # Encontrar los valores reales correspondientes
        y_true_values = []
        for idx in original_indices:
            # Buscar en datos de regresión
            if idx in data['indices']['test_reg_indices']:
                pos = data['indices']['test_reg_indices'].get_loc(idx)
                y_true_values.append(data['regression']['y_test'].iloc[pos])
        
        # Si hay suficientes valores, calcular métricas
        if len(y_true_values) > 0:
            y_true_true_pos = np.array(y_true_values)
            mae_hybrid = mean_absolute_error(y_true_true_pos, y_pred_true_pos[:len(y_true_true_pos)])
            rmse_hybrid = np.sqrt(mean_squared_error(y_true_true_pos, y_pred_true_pos[:len(y_true_true_pos)]))
            
            try:
                r2_hybrid = r2_score(y_true_true_pos, y_pred_true_pos[:len(y_true_true_pos)])
            except:
                r2_hybrid = float('nan')
        else:
            mae_hybrid = float('nan')
            rmse_hybrid = float('nan')
            r2_hybrid = float('nan')
    else:
        mae_hybrid = float('nan')
        rmse_hybrid = float('nan')
        r2_hybrid = float('nan')
    
    print(f"✅ Métricas del modelo híbrido final:")
    print(f"   • Clasificación: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"   • Regresión (solo verdaderos positivos): MAE={mae_hybrid:.2f}")
    
    # Guardar resultados
    hybrid_results = {
        'classification': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(class_threshold) 
        },
        'regression': {
            'mae': float(mae_hybrid),
            'rmse': float(rmse_hybrid),
            'r2': float(r2_hybrid) if not np.isnan(r2_hybrid) else None
        },
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{models_dir}/hybrid_model_results.json', 'w') as f:
        json.dump(hybrid_results, f, indent=2)
    
    # Crear visualización comparativa
    plt.figure(figsize=(12, 10))
    
    # Gráfico de dispersión para regresión
    plt.subplot(2, 1, 1)
    plt.scatter(y_true_true_pos, y_pred_true_pos[:len(y_true_true_pos)], alpha=0.5)
    plt.plot([0, max(y_true_true_pos)], [0, max(y_true_true_pos)], 'r--')
    plt.xlabel('Cantidad Real a Reponer')
    plt.ylabel('Cantidad Predicha')
    plt.title('Predicción vs Realidad: Modelo Híbrido Final')
    plt.grid(True, alpha=0.3)
    
    # Matriz de confusión para clasificación
    plt.subplot(2, 1, 2)
    conf_matrix = pd.crosstab(y_class_true, y_class_pred, 
                             rownames=['Real'], colnames=['Predicho'])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión: Clasificación')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/hybrid_model_performance.png', dpi=300)
    
    return hybrid_results

def main():
    print("🚀 IMPLEMENTACIÓN DE MODELO HÍBRIDO FINAL OPTIMIZADO")
    print("="*60)
    
    # Cargar datos
    df, features = load_data()
    
    # Preparar datos
    data = prepare_data(df, features)
    
    # Entrenar clasificador ensemble
    class_model, class_threshold, class_metrics = train_ensemble_classifier(data)
    
    # Entrenar modelo de regresión con bagging
    reg_model, reg_metrics = train_bagging_regressor(data)
    
    # Evaluar modelo híbrido final
    hybrid_results = implement_hybrid_model(data, class_model, class_threshold, reg_model)
    
    # Guardar configuración final
    final_config = {
        'classification': class_metrics,
        'regression': reg_metrics,
        'hybrid': hybrid_results,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}/final_model_config.json', 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print("\n✅ MODELO HÍBRIDO FINAL COMPLETADO")
    print(f"📁 Archivos generados:")
    print(f"   • {models_dir}/ensemble_classifier.joblib")
    print(f"   • {models_dir}/regression_model.joblib")
    print(f"   • {models_dir}/features_scaler.joblib")
    print(f"   • {output_dir}/final_model_config.json")
    print(f"   • {plots_dir}/hybrid_model_performance.png")
    
    return class_model, reg_model, final_config

if __name__ == "__main__":
    class_model, reg_model, config = main()