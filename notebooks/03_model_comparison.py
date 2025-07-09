# 03_model_comparison.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
   classification_report, confusion_matrix, roc_curve,
   mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
from src.utils.feature_utils import remove_leaky_features, verify_no_leakage
import warnings
import json
from datetime import datetime
from lightgbm.sklearn import LGBMRegressor
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
# Algoritmos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

# Algoritmos de regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

output_dir = 'results/03_model_comparison'
plots_dir = f'{output_dir}/plots'
models_dir = 'models'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_engineered_data():
   """Cargar datos con feature engineering"""
   try:
       df = pd.read_csv('data/processed/02_features/features_engineered.csv')
       print(f"✅ Datos cargados: {df.shape}")
       
       # Cargar metadata si existe
       try:
           with open('results/02_feature_engineering/feature_metadata.json', 'r') as f:
               metadata = json.load(f)
               features = metadata.get('features', [])
       except (FileNotFoundError, json.JSONDecodeError):
           # Si no hay metadata, inferir features
           features = [col for col in df.columns 
                       if col not in ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                                    'necesita_reposicion', 'cantidad_a_reponer', 
                                    'log_cantidad_a_reponer']]
       
       print(f"✅ Features identificadas: {len(features)}")
       return df, features

   except FileNotFoundError:
       print("❌ Error: Ejecuta primero 02_feature_engineering.py")
       return None, None

def prepare_ml_datasets(df, features):
    """Preparar datasets para clasificación y regresión SIN LEAKAGE"""
    print("\n📊 PREPARANDO DATASETS PARA ML")
    print("-" * 30)
    
    # Verificación final de leakage
    try:
        X_clean = df[features].copy()
        print("✅ Usando features desde metadata para evitar leakage")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        # Filtrado manual
        leaky_patterns = ['stock_', 'recuent', 'ocupacion', 'gap_']
        clean_features = [f for f in features 
                         if not any(pattern in f.lower() for pattern in leaky_patterns)]
        X_clean = df[clean_features].copy()
        
        print(f"🗑️ Features con leakage eliminados: {len(features) - len(clean_features)}")
    
    # Seleccionar solo columnas numéricas para evitar errores con fechas
    numeric_cols = X_clean.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < len(X_clean.columns):
        print(f"⚠️ Se eliminaron {len(X_clean.columns) - len(numeric_cols)} columnas no numéricas")
        X_clean = X_clean[numeric_cols]
    
    X = X_clean.copy()
    
    # Manejar todos los NaN antes de entrenar modelos
    nan_cols = X.columns[X.isna().sum() > 0]
    if len(nan_cols) > 0:
        print(f"⚠️ Detectados NaN en {len(nan_cols)} columnas - aplicando imputación")
        for col in X.columns:
            if X[col].isna().any():
                # Usar mediana para imputación 
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"   • Imputados {X[col].isna().sum()} valores en {col} con mediana: {median_val}")
    
    # Verificar que no queden NaN
    if X.isna().any().any():
        print("⚠️ Aún hay NaN en el dataset después de la imputación")
        print(f"   • Columnas con NaN: {X.columns[X.isna().any()].tolist()}")
        # Imputación forzada con 0
        X = X.fillna(0)
        print("   • Se han imputado los NaN restantes con 0")
    
    # Targets
    y_classification = df['necesita_reposicion'].copy() if 'necesita_reposicion' in df.columns else None
    y_regression = df['cantidad_a_reponer'].copy() if 'cantidad_a_reponer' in df.columns else None
    y_regression_log = df['log_cantidad_a_reponer'].copy() if 'log_cantidad_a_reponer' in df.columns else None
    
    # Si no existe el target log, pero existe el target normal, crear la versión log
    if y_regression_log is None and y_regression is not None:
        y_regression_log = np.log1p(y_regression)
    
    print(f"\n✅ Dataset preparado:")
    print(f"   • Features finales: {X.shape[1]}")
    print(f"   • Registros: {X.shape[0]:,}")
    
    if y_classification is not None:
        print(f"   • Clasificación - Balance: {y_classification.mean():.1%} positivos")
    
    if y_regression is not None:
        print(f"   • Regresión - Media: {y_regression.mean():.1f}, Mediana: {y_regression.median():.1f}")
        print(f"   • Casos que necesitan reposición: {(y_regression > 0).sum():,} ({(y_regression > 0).mean():.1%})")
    
    return X, y_classification, y_regression, y_regression_log, numeric_cols

def compare_classification_algorithms(X, y, df):
   """Comparar algoritmos de clasificación"""
   print("\n🎯 COMPARACIÓN - ALGORITMOS DE CLASIFICACIÓN")
   print("="*50)
   
   if y is None:
       print("⚠️ No se encontró el target de clasificación 'necesita_reposicion'")
       return {}, {}, None
   
   # Split inicial con estratificación
   X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
   
   print(f"📊 Split: Train {X_train.shape[0]:,}, Test {X_test.shape[0]:,}")
   print(f"📊 Balance train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
   
   # Escalado robusto
   scaler = RobustScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Definir algoritmos
   algorithms = {
       'Logistic Regression': LogisticRegression(
           random_state=42, max_iter=1000, C=1.0, class_weight='balanced'),
       'Random Forest': RandomForestClassifier(
           n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'),
       'Gradient Boosting': GradientBoostingClassifier(
           n_estimators=100, max_depth=5, random_state=42),
       'XGBoost': xgb.XGBClassifier(
           n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss', 
           n_jobs=-1, scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1]),
       'LightGBM': lgb.LGBMClassifier(
           n_estimators=100, max_depth=5, random_state=42, verbose=-1, 
           n_jobs=-1, class_weight='balanced'),
       'SVM': SVC(
           random_state=42, probability=True, C=1.0, class_weight='balanced'),
       'K-Nearest Neighbors': KNeighborsClassifier(
           n_neighbors=5, n_jobs=-1, weights='distance'),
       'Naive Bayes': GaussianNB()
   }
   
   # Algoritmos que necesitan escalado
   scaled_algorithms = ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']
   
   results = {}
   predictions = {}
   
   # Cross-validation estratificada
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
   for name, model in algorithms.items():
       print(f"\n🔍 Evaluando {name}...")
       
       # Seleccionar datos
       if name in scaled_algorithms:
           X_tr, X_te = X_train_scaled, X_test_scaled
       else:
           X_tr, X_te = X_train, X_test
       
       try:
           # Cross-validation
           cv_scores = cross_val_score(
               model, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
           cv_roc = cross_val_score(
               model, X_tr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
           
           # Entrenar modelo final
           model.fit(X_tr, y_train)
           
           # Predicciones
           y_pred = model.predict(X_te)
           y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
           
           # Métricas
           accuracy = accuracy_score(y_test, y_pred)
           precision = precision_score(y_test, y_pred, average='binary')
           recall = recall_score(y_test, y_pred, average='binary')
           f1 = f1_score(y_test, y_pred, average='binary')
           roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
           
           results[name] = {
               'CV_Accuracy_Mean': float(cv_scores.mean()),
               'CV_Accuracy_Std': float(cv_scores.std()),
               'CV_ROC_AUC_Mean': float(cv_roc.mean()),
               'CV_ROC_AUC_Std': float(cv_roc.std()),
               'Test_Accuracy': float(accuracy),
               'Test_Precision': float(precision),
               'Test_Recall': float(recall),
               'Test_F1': float(f1),
               'Test_ROC_AUC': float(roc_auc) if roc_auc is not None else None
           }
           
           predictions[name] = {
               'y_pred': y_pred,
               'y_proba': y_proba,
               'model': model
           }
           
           print(f"   ✅ CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
           print(f"   ✅ Test Accuracy: {accuracy:.3f}")
           print(f"   ✅ Test ROC-AUC: {roc_auc:.3f}" if roc_auc else "   ⚠️ ROC-AUC: N/A")
           
       except Exception as e:
           print(f"   ❌ Error: {str(e)}")
           continue
   
   return results, predictions, (X_test, y_test, scaler)

def evaluate_model(model, X_test, y_test_log):
    """Evalúa modelo de regresión con transformación logarítmica de forma consistente"""
    # Predecir en escala logarítmica
    y_pred_log = model.predict(X_test)
    
    # Convertir a escala original para métricas
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)
    
    # Calcular métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return mae, rmse, r2, y_pred

def compare_regression_algorithms(X, y, y_log, df):
  """Comparar algoritmos de regresión"""
  print("\n📈 COMPARACIÓN - ALGORITMOS DE REGRESIÓN")
  print("="*50)
  
  if y is None:
      print("⚠️ No se encontró el target de regresión 'cantidad_a_reponer'")
      return {}, {}, None
  
  # Filtrar para casos que necesitan reposición
  mask = y > 0
  X_reg = X[mask].copy()
  y_reg = y[mask].copy()
  y_reg_log = y_log[mask].copy() if y_log is not None else None
  
  print(f"📊 Datos para regresión: {len(X_reg):,} registros (casos con reposición > 0)")
  
  if len(X_reg) < 100:
      print("⚠️ Pocos datos para regresión. Usando todos los datos.")
      X_reg, y_reg = X, y
      y_reg_log = y_log
  
  # Split para regresión
  X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
    )
  y_train_log = y_reg_log.loc[X_train.index] if y_reg_log is not None else None
  y_test_log = y_reg_log.loc[X_test.index] if y_reg_log is not None else None
  
  # Escalar
  scaler = RobustScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Definir algoritmos
  algorithms = {
      'Linear Regression': LinearRegression(),
      'Ridge Regression': Ridge(alpha=1.0),
      'Lasso Regression': Lasso(alpha=0.1),
      'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
      'Random Forest': RandomForestRegressor(
          n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
      'Gradient Boosting': GradientBoostingRegressor(
          n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
      'XGBoost': xgb.XGBRegressor(
          n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
      'LightGBM': lgb.LGBMRegressor(
          n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1),
      'SVR': SVR(C=1.0, gamma='scale'),
      'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
  }
  
  results = {}
  predictions = {}
  
  # Evaluar con y sin transformación logarítmica
  for log_transform in [True, False]:
      if log_transform and y_train_log is None:
          continue
          
      current_y_train = y_train_log if log_transform else y_train
      current_y_test = y_test_log if log_transform else y_test
      model_suffix = " (Log)" if log_transform else ""
      
      print(f"\n🔄 Evaluando con transformación logarítmica: {log_transform}")
      
      for name, model in algorithms.items():
          print(f"\n🔍 Evaluando {name}{model_suffix}...")
          
          # Usar siempre datos escalados para consistencia
          X_tr, X_te = X_train_scaled, X_test_scaled
          
          try:
              # Cross-validation
              cv_scores = cross_val_score(
                  model, X_tr, current_y_train, cv=5, 
                  scoring='neg_mean_absolute_error', n_jobs=-1)
              cv_r2 = cross_val_score(
                  model, X_tr, current_y_train, cv=5, 
                  scoring='r2', n_jobs=-1)
              
              # Entrenar modelo
              model.fit(X_tr, current_y_train)
              
              # Predicciones
              y_pred = model.predict(X_te)
              
              # Si se usó transformación logarítmica, volver a escala original
              if log_transform:
                  y_pred_original = np.expm1(y_pred)
                  y_true_original = y_test
              else:
                  y_pred_original = y_pred
                  y_true_original = y_test
              
              # Asegurar predicciones no negativas
              y_pred_original = np.maximum(0, y_pred_original)
              
              # Métricas
              mae = mean_absolute_error(y_true_original, y_pred_original)
              rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
              r2 = r2_score(y_true_original, y_pred_original)
              
              # SMAPE (más estable para valores cercanos a cero)
              epsilon = 1
              smape = 100 * np.mean(2 * np.abs(y_true_original - y_pred_original) / 
                                  (np.abs(y_true_original) + np.abs(y_pred_original) + epsilon))
              
              # Guardar resultados
              algorithm_name = f"{name}{model_suffix}"
              results[algorithm_name] = {
                  'CV_MAE_Mean': float(-cv_scores.mean()),
                  'CV_MAE_Std': float(cv_scores.std()),
                  'CV_R2_Mean': float(cv_r2.mean()),
                  'CV_R2_Std': float(cv_r2.std()),
                  'Test_MAE': float(mae),
                  'Test_RMSE': float(rmse),
                  'Test_R2': float(r2),
                  'Test_SMAPE': float(smape)
              }
              
              predictions[algorithm_name] = {
                  'y_pred': y_pred_original,
                  'model': model,
                  'log_transform': log_transform
              }
              
              print(f"   ✅ CV MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
              print(f"   ✅ Test MAE: {mae:.2f}")
              print(f"   ✅ Test R²: {r2:.3f}")
              
          except Exception as e:
              print(f"   ❌ Error: {str(e)}")
              continue
  
  return results, predictions, (X_test, y_test, scaler)

def create_results_visualizations(class_results, reg_results):
   """Crear visualizaciones de resultados"""
   print("\n📊 CREANDO VISUALIZACIONES")
   print("-" * 30)
   
   # Visualizaciones de clasificación
   if class_results:
       plt.figure(figsize=(12, 8))
       
       # Convertir a DataFrame
       class_df = pd.DataFrame({k: v['Test_Accuracy'] for k, v in class_results.items()}, 
                              index=['Accuracy']).T
       class_df['F1-Score'] = pd.Series({k: v['Test_F1'] for k, v in class_results.items()})
       class_df['ROC-AUC'] = pd.Series({k: v['Test_ROC_AUC'] for k, v in class_results.items() 
                                       if v.get('Test_ROC_AUC') is not None})
       
       # Ordenar por Accuracy
       class_df = class_df.sort_values('Accuracy', ascending=False)
       
       # Graficar
       ax = class_df.plot(kind='bar', figsize=(12, 6))
       plt.title('Métricas de Clasificación por Algoritmo')
       plt.ylabel('Puntuación')
       plt.grid(axis='y', alpha=0.3)
       plt.legend(loc='lower right')
       plt.tight_layout()
       plt.savefig(f'{plots_dir}/classification_metrics.png', dpi=300)
   
   # Visualizaciones de regresión
   if reg_results:
       # Separar resultados con y sin transformación logarítmica
       log_results = {k: v for k, v in reg_results.items() if '(Log)' in k}
       normal_results = {k: v for k, v in reg_results.items() if '(Log)' not in k}
       
       # Procesar cada conjunto
       for result_set, title_suffix in [(log_results, 'Transformación Logarítmica'), 
                                       (normal_results, 'Sin Transformación')]:
           if not result_set:
               continue
               
           plt.figure(figsize=(12, 8))
           
           # Crear DataFrame con R²
           r2_df = pd.DataFrame({k: v['Test_R2'] for k, v in result_set.items()}, 
                               index=['R²']).T
           
           # Ordenar por R²
           r2_df = r2_df.sort_values('R²', ascending=False)
           
           # Graficar
           ax = r2_df.plot(kind='bar', figsize=(12, 6), color='purple')
           plt.title(f'R² por Algoritmo - {title_suffix}')
           plt.ylabel('R² Score')
           plt.grid(axis='y', alpha=0.3)
           plt.tight_layout()
           plt.savefig(f'{plots_dir}/regression_r2_{title_suffix.replace(" ", "_").lower()}.png', 
                      dpi=300)
           
           # Gráfico para MAE
           plt.figure(figsize=(12, 8))
           mae_df = pd.DataFrame({k: v['Test_MAE'] for k, v in result_set.items()}, 
                                index=['MAE']).T
           
           # Ordenar por MAE (ascendente, menor es mejor)
           mae_df = mae_df.sort_values('MAE', ascending=True)
           
           # Graficar
           ax = mae_df.plot(kind='bar', figsize=(12, 6), color='orange')
           plt.title(f'MAE por Algoritmo - {title_suffix}')
           plt.ylabel('Error Absoluto Medio')
           plt.grid(axis='y', alpha=0.3)
           plt.tight_layout()
           plt.savefig(f'{plots_dir}/regression_mae_{title_suffix.replace(" ", "_").lower()}.png', 
                      dpi=300)
   
   print("✅ Visualizaciones guardadas en 'results/plots/'")

def save_results(class_results, reg_results):
   """Guardar resultados en CSV"""
   print("\n💾 GUARDANDO RESULTADOS")
   print("-" * 30)
   
   # Guardar resultados de clasificación
   if class_results:
       class_df = pd.DataFrame(class_results).T
       class_df.to_csv(f'{output_dir}/classification_results.csv')
       print(f"✅ Clasificación guardada: results/classification_results.csv")
   
   # Guardar resultados de regresión
   if reg_results:
       reg_df = pd.DataFrame(reg_results).T
       reg_df.to_csv(f'{output_dir}/regression_results.csv')
       print(f"✅ Regresión guardada: results/regression_results.csv")

def save_importance_analysis(predictions, features_used):
   """Guardar análisis de importancia de variables para modelos basados en árboles"""
   print("\n📊 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
   print("-" * 30)
   
   # Modelos basados en árboles
   tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 
                 'Random Forest (Log)', 'Gradient Boosting (Log)', 'XGBoost (Log)', 'LightGBM (Log)']
   
   # Verificar disponibilidad
   available_models = [name for name in tree_models 
                      if name in predictions and hasattr(predictions[name]['model'], 'feature_importances_')]
   
   if not available_models:
       print("⚠️ No se encontraron modelos basados en árboles")
       return
   
   # Analizar importancia para cada modelo
   importances_data = {}
   
   for model_name in available_models:
       model = predictions[model_name]['model']
       importances = model.feature_importances_
       
       # Verificar que tenemos la misma cantidad de features
       if len(importances) != len(features_used):
           print(f"⚠️ Error: El modelo {model_name} tiene {len(importances)} importancias " 
                 f"pero hay {len(features_used)} features")
           continue
       
       # Crear DataFrame con importancias
       importance_df = pd.DataFrame({
           'feature': features_used,
           'importance': importances
       }).sort_values('importance', ascending=False)
       
       # Guardar importancias
       importances_data[model_name] = importance_df
       
       # Visualizar top 20 características
       plt.figure(figsize=(12, 8))
       top_features = importance_df.head(20)
       sns.barplot(x='importance', y='feature', data=top_features)
       plt.title(f'Top 20 Características Importantes - {model_name}')
       plt.xlabel('Importancia')
       plt.tight_layout()
       plt.savefig(f'{plots_dir}/importance_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png', 
                  dpi=300)
       
       # Guardar en CSV
       importance_df.to_csv(f'{output_dir}/importance_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.csv', 
                           index=False)
       
       print(f"✅ Análisis de importancia para {model_name} guardado")
       print(f"   • Top 5 características:")
       for i, row in top_features.head(5).iterrows():
           print(f"     - {row['feature']}: {row['importance']:.4f}")
   
   # Crear visualización comparativa de top features
   if len(available_models) > 1:
       # Obtener top 15 características (unión de top 10 de cada modelo)
       top_features_all = set()
       for model_name in available_models:
           if model_name in importances_data:
               top_10 = importances_data[model_name].head(10)['feature'].tolist()
               top_features_all.update(top_10)
       
       # Crear DataFrame para comparación
       comparison_data = []
       for feature in top_features_all:
           for model_name in available_models:
               if model_name not in importances_data:
                   continue
                   
               # Encontrar la importancia de esta característica
               df = importances_data[model_name]
               importance_value = df.loc[df['feature'] == feature, 'importance'].values[0] if feature in df['feature'].values else 0
               
               comparison_data.append({
                   'feature': feature,
                   'model': model_name,
                   'importance': importance_value
               })
       
       comparison_df = pd.DataFrame(comparison_data)
       
       # Visualizar comparación
       plt.figure(figsize=(16, 12))
       sns.barplot(x='importance', y='feature', hue='model', data=comparison_df)
       plt.title('Comparación de Importancia de Características entre Modelos')
       plt.xlabel('Importancia')
       plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.tight_layout()
       plt.savefig(f'{plots_dir}/importance_comparison.png', dpi=300)
       
       print(f"✅ Comparación de importancia entre modelos guardada")

def print_summary(class_results, reg_results):
   """Imprimir resumen de resultados"""
   print("\n🏆 RESUMEN DE RESULTADOS")
   print("="*50)
   
   if class_results:
       print("\n🎯 CLASIFICACIÓN (¿Necesita reposición?):")
       
       # Convertir a DataFrame
       class_df = pd.DataFrame(class_results).T
       
       # Mejores por métrica
       best_accuracy = class_df['Test_Accuracy'].idxmax()
       best_f1 = class_df['Test_F1'].idxmax()
       best_roc = class_df['Test_ROC_AUC'].idxmax() if 'Test_ROC_AUC' in class_df.columns else None
       
       print(f"   🥇 Mejor Accuracy: {best_accuracy} ({class_df.loc[best_accuracy, 'Test_Accuracy']:.3f})")
       print(f"   🥇 Mejor F1-Score: {best_f1} ({class_df.loc[best_f1, 'Test_F1']:.3f})")
       if best_roc:
           print(f"   🥇 Mejor ROC-AUC: {best_roc} ({class_df.loc[best_roc, 'Test_ROC_AUC']:.3f})")
       
       # Top 3 por accuracy
       print(f"\n   📊 Top 3 por Accuracy:")
       top_3_acc = class_df.sort_values('Test_Accuracy', ascending=False).head(3)
       for i, (name, row) in enumerate(top_3_acc.iterrows(), 1):
           print(f"      {i}. {name}: {row['Test_Accuracy']:.3f}")
   
   if reg_results:
       print("\n📈 REGRESIÓN (¿Cuánto reponer?):")
       
       # Separar resultados con y sin transformación logarítmica
       log_results = {k: v for k, v in reg_results.items() if '(Log)' in k}
       normal_results = {k: v for k, v in reg_results.items() if '(Log)' not in k}
       
       for result_set, title in [(normal_results, "Sin transformación"), 
                                (log_results, "Con transformación logarítmica")]:
           if not result_set:
               continue
           
           print(f"\n   🔄 {title}:")
           
           # Convertir a DataFrame
           reg_df = pd.DataFrame(result_set).T
           
           # Mejores por métrica
           best_mae = reg_df['Test_MAE'].idxmin()  # Menor es mejor
           best_r2 = reg_df['Test_R2'].idxmax()    # Mayor es mejor
           
           print(f"   🥇 Mejor MAE: {best_mae} ({reg_df.loc[best_mae, 'Test_MAE']:.2f})")
           print(f"   🥇 Mejor R²: {best_r2} ({reg_df.loc[best_r2, 'Test_R2']:.3f})")
           
           # Top 3 por R²
           print(f"\n   📊 Top 3 por R²:")
           top_3_r2 = reg_df.sort_values('Test_R2', ascending=False).head(3)
           for i, (name, row) in enumerate(top_3_r2.iterrows(), 1):
               print(f"      {i}. {name}: {row['Test_R2']:.3f}")

def save_best_models(class_predictions, reg_predictions):
   """Guardar los mejores modelos para su posterior uso"""
   print("\n💾 GUARDANDO MEJORES MODELOS")
   print("-" * 30)
   
   # Mejor modelo de clasificación
   if class_predictions:
       try:
           from joblib import dump
           
           # Identificar mejor modelo
           best_model_name = None
           best_f1 = -1
           best_test_data = None
           
           for name, pred_data in class_predictions.items():
               # Preferir modelos con buena interpretabilidad
               if 'Random Forest' in name or 'XGBoost' in name or 'LightGBM' in name:
                   model = pred_data['model']
                   y_pred = pred_data['y_pred']
                   y_true = None
                   
                   # Intentar obtener y_test de los datos de prueba almacenados
                   for test_data_key, test_data_val in locals().items():
                       if (test_data_key.endswith('test_data') and 
                           isinstance(test_data_val, tuple) and 
                           len(test_data_val) > 1):
                           _, y_true, _ = test_data_val
                           best_test_data = test_data_val
                           break
                   
                   if y_true is not None:
                       f1 = f1_score(y_true, y_pred)
                       if f1 > best_f1:
                           best_f1 = f1
                           best_model_name = name
           
           if best_model_name:
               best_model = class_predictions[best_model_name]['model']
               dump(best_model, f'{models_dir}/best_classification_model.joblib')
               
               # Guardar metadata
               model_meta = {
                   'model_name': best_model_name,
                   'f1_score': float(best_f1),
                   'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
               }
               
               with open(f'{models_dir}/classification_model_meta.json', 'w') as f:
                   json.dump(model_meta, f, indent=2)
               
               print(f"✅ Mejor modelo de clasificación guardado: {best_model_name}")
       except Exception as e:
           print(f"⚠️ Error al guardar modelo de clasificación: {e}")
   
   # Mejor modelo de regresión
   if reg_predictions:
       try:
           from joblib import dump
           
           # Identificar mejor modelo
           best_model_name = None
           best_r2 = -float('inf')
           
           for name, pred_data in reg_predictions.items():
               # Preferir modelos con buena interpretabilidad y sin log transform
               if ('Random Forest' in name or 'XGBoost' in name or 'LightGBM' in name) and '(Log)' not in name:
                   model = pred_data['model']
                   y_pred = pred_data['y_pred']
                   
                   # Intentar calcular R² directamente si tenemos los datos
                   r2 = -float('inf')
                   for test_data_key, test_data_val in locals().items():
                       if (test_data_key.endswith('test_data') and 
                           isinstance(test_data_val, tuple) and 
                           len(test_data_val) > 1):
                           _, y_true, _ = test_data_val
                           if len(y_pred) == len(y_true):
                               try:
                                   r2 = r2_score(y_true, y_pred)
                               except:
                                   pass
                               break
                   
                   if r2 > best_r2:
                       best_r2 = r2
                       best_model_name = name
           
           if best_model_name:
               best_model = reg_predictions[best_model_name]['model']
               dump(best_model, f'{models_dir}/best_regression_model.joblib')
               
               # Guardar metadata
               model_meta = {
                   'model_name': best_model_name,
                   'r2_score': float(best_r2),
                   'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
               }
               
               with open(f'{models_dir}/regression_model_meta.json', 'w') as f:
                   json.dump(model_meta, f, indent=2)
               
               print(f"✅ Mejor modelo de regresión guardado: {best_model_name}")
       except Exception as e:
           print(f"⚠️ Error al guardar modelo de regresión: {e}")

def main():
   print("🚀 INICIANDO COMPARACIÓN DE ALGORITMOS ML")
   print("="*60)
   
   # Cargar datos
   df, features = load_engineered_data()
   if df is None:
       return None, None, None
   
   # Preparar datasets
   X, y_class, y_reg, y_reg_log, final_features = prepare_ml_datasets(df, features)
   
   # Comparar algoritmos de clasificación
   class_results, class_predictions, class_test_data = compare_classification_algorithms(X, y_class, df)
   
   # Comparar algoritmos de regresión
   reg_results, reg_predictions, reg_test_data = compare_regression_algorithms(X, y_reg, y_reg_log, df)
   
   # Crear visualizaciones
   create_results_visualizations(class_results, reg_results)
   
   # Guardar resultados
   save_results(class_results, reg_results)
   
   # Guardar análisis de importancia
   if class_predictions:
       save_importance_analysis(class_predictions, final_features)
   
   # Guardar mejores modelos
   save_best_models(class_predictions, reg_predictions)
   
   # Imprimir resumen
   print_summary(class_results, reg_results)
   
   print(f"\n✅ COMPARACIÓN COMPLETADA")
   print(f"📁 Archivos generados:")
   print(f"   • results/classification_results.csv")
   print(f"   • results/regression_results.csv") 
   print(f"   • results/plots/ (múltiples gráficos)")
   print(f"   • models/ (mejores modelos)")
   
   return class_results, reg_results, final_features

if __name__ == "__main__":
   class_results, reg_results, features = main()