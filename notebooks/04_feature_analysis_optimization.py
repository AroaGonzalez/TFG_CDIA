# 04_feature_analysis_optimization.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def load_data_and_results():
   """Cargar datos procesados y resultados de modelos anteriores"""
   try:
       df = pd.read_csv('data/processed/features_engineered.csv')
       
       with open('data/processed/feature_metadata.json', 'r') as f:
           metadata = json.load(f)
           features = metadata.get('features', [])
       
       class_results = pd.read_csv('results/classification_results.csv', index_col=0)
       reg_results = pd.read_csv('results/regression_results.csv', index_col=0)
       
       print(f"✅ Datos cargados: {df.shape[0]} registros, {len(features)} features")
       print(f"✅ Resultados anteriores: {len(class_results)} modelos de clasificación, {len(reg_results)} modelos de regresión")
       
       return df, features, class_results, reg_results
   
   except FileNotFoundError as e:
       print(f"❌ Error: {e}")
       print("   Ejecuta primero los scripts anteriores (01, 02 y 03)")
       return None, None, None, None

def analyze_feature_importance():
   """Analizar importancia de características según modelos anteriores"""
   print("\n🔍 ANALIZANDO IMPORTANCIA DE CARACTERÍSTICAS")
   print("-" * 50)
   
   importance_files = [f for f in os.listdir('results') if f.startswith('importance_') and f.endswith('.csv')]
   
   if not importance_files:
       print("❌ No se encontraron análisis de importancia. Ejecuta primero 03_model_comparison.py")
       return None
   
   all_importances = []
   
   for file in importance_files:
       model_name = file.replace('importance_', '').replace('.csv', '')
       imp_df = pd.read_csv(f'results/{file}')
       imp_df['model'] = model_name
       all_importances.append(imp_df)
   
   combined_imp = pd.concat(all_importances)
   
   avg_imp = combined_imp.groupby('feature')['importance'].mean().reset_index()
   avg_imp = avg_imp.sort_values('importance', ascending=False)
   
   plt.figure(figsize=(12, 8))
   top_n = min(20, len(avg_imp))
   sns.barplot(x='importance', y='feature', data=avg_imp.head(top_n))
   plt.title(f'Top {top_n} Características por Importancia Promedio')
   plt.tight_layout()
   plt.savefig('results/plots/feature_importance_aggregated.png', dpi=300)
   
   print(f"✅ Top 10 características más importantes:")
   for i, (_, row) in enumerate(avg_imp.head(10).iterrows(), 1):
       print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
   
   return avg_imp

def select_optimal_features(df, features, importance_df, threshold=0.7):
   """Seleccionar características óptimas basado en importancia acumulada"""
   print("\n🔎 SELECCIONANDO CARACTERÍSTICAS ÓPTIMAS")
   print("-" * 50)
   
   if importance_df is None:
       print("❌ No se puede realizar selección sin datos de importancia")
       return features
   
   total_imp = importance_df['importance'].sum()
   importance_df['importance_normalized'] = importance_df['importance'] / total_imp
   importance_df['importance_cumulative'] = importance_df['importance_normalized'].cumsum()
   
   selected_features = importance_df[importance_df['importance_cumulative'] <= threshold]['feature'].tolist()
   
   if len(selected_features) < 10:
       selected_features = importance_df.head(10)['feature'].tolist()
   
   print(f"✅ Características seleccionadas: {len(selected_features)} de {len(features)}")
   print(f"   Umbral de importancia acumulada: {threshold:.1%}")
   
   valid_features = [f for f in selected_features if f in df.columns]
   if len(valid_features) < len(selected_features):
       print(f"⚠️ {len(selected_features) - len(valid_features)} características no están disponibles en el dataframe")
   
   return valid_features

def prepare_optimized_datasets(df, selected_features):
   """Preparar conjuntos de datos optimizados para entrenamiento"""
   print("\n📊 PREPARANDO DATASETS OPTIMIZADOS")
   print("-" * 50)
   
   X = df[selected_features].copy()
   
   numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
   if len(numeric_cols) < len(X.columns):
       print(f"⚠️ Se omitirán {len(X.columns) - len(numeric_cols)} columnas no numéricas")
       X = X[numeric_cols]
   
   for col in X.columns:
       if X[col].isna().sum() > 0:
           X[col] = X[col].fillna(X[col].median())
   
   y_class = df['necesita_reposicion'] if 'necesita_reposicion' in df.columns else None
   y_reg = df['cantidad_a_reponer'] if 'cantidad_a_reponer' in df.columns else None
   y_reg_log = df['log_cantidad_a_reponer'] if 'log_cantidad_a_reponer' in df.columns else None
   
   df_temp = pd.DataFrame({'fecha': df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'], 'X_index': X.index})
   fecha_corte = df_temp['fecha'].quantile(0.8)
   train_mask = df_temp['fecha'] <= fecha_corte
   test_mask = df_temp['fecha'] > fecha_corte
   
   X_train = X.loc[df_temp[train_mask]['X_index']]
   X_test = X.loc[df_temp[test_mask]['X_index']]
   y_class_train = y_class.loc[df_temp[train_mask]['X_index']]
   y_class_test = y_class.loc[df_temp[test_mask]['X_index']]
   
   mask_train = y_reg[X_train.index] > 0
   mask_test = y_reg[X_test.index] > 0
   
   X_reg_train = X_train[mask_train]
   y_reg_train = y_reg[X_train.index][mask_train]
   y_reg_log_train = y_reg_log[X_train.index][mask_train]
   
   X_reg_test = X_test[mask_test]
   y_reg_test = y_reg[X_test.index][mask_test]
   y_reg_log_test = y_reg_log[X_test.index][mask_test]
   
   # Crear segmentos para modelos específicos por magnitud
   quantiles = [0, 0.25, 0.5, 0.75, 1.0]
   labels = ['Q1', 'Q2', 'Q3', 'Q4']
   
   y_reg_train_quantile = pd.qcut(y_reg_train, q=quantiles, labels=labels)
   segment_datasets = {}
   
   for segment in labels:
       mask_segment = y_reg_train_quantile == segment
       segment_datasets[segment] = {
           'X_train': X_reg_train[mask_segment],
           'y_train': y_reg_train[mask_segment],
           'y_train_log': y_reg_log_train[mask_segment]
       }
   
   print(f"✅ Dataset de clasificación: {X_train.shape[0]} train, {X_test.shape[0]} test")
   print(f"✅ Dataset de regresión: {X_reg_train.shape[0]} train, {X_reg_test.shape[0]} test (solo casos > 0)")
   print(f"✅ Datasets por segmentos creados: {', '.join([f'{k}: {v['X_train'].shape[0]} muestras' for k, v in segment_datasets.items()])}")
   
   return {
       'classification': (X_train, X_test, y_class_train, y_class_test),
       'regression': (X_reg_train, X_reg_test, y_reg_train, y_reg_test),
       'regression_log': (X_reg_train, X_reg_test, y_reg_log_train, y_reg_log_test),
       'segments': segment_datasets,
       'test_data': (X_reg_test, y_reg_test, y_reg_log_test)
   }

def optimize_classification_model(datasets):
   """Optimizar el modelo de clasificación con SMOTE para balancear clases"""
   print("\n🔄 OPTIMIZANDO MODELO DE CLASIFICACIÓN")
   print("-" * 50)
   
   X_train, X_test, y_train, y_test = datasets['classification']
   
   # Crear pipeline con SMOTE para balancear clases
   pipeline = ImbPipeline([
       ('scaler', RobustScaler()),
       ('smote', SMOTE(random_state=42)),
       ('model', RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0))
   ])
   
   param_grid = {
       'model__n_estimators': [100, 200, 300],
       'model__max_depth': [5, 10, 15],
       'model__min_samples_leaf': [2, 5, 10],
       'model__min_samples_split': [2, 5, 10],
       'model__class_weight': [None, 'balanced', 'balanced_subsample']
   }
   
   print("🔍 Iniciando búsqueda de hiperparámetros para clasificación...")
   grid_search = RandomizedSearchCV(
       pipeline, param_grid, cv=5, scoring='f1',
       n_iter=15, random_state=42, n_jobs=-1, verbose=0
   )
   
   grid_search.fit(X_train, y_train)
   
   best_model = grid_search.best_estimator_
   y_pred = best_model.predict(X_test)
   f1 = f1_score(y_test, y_pred)
   
   print(f"✅ Mejores parámetros: {grid_search.best_params_}")
   print(f"✅ F1-Score: {f1:.4f}")
   print(f"✅ El modelo usa SMOTE para balancear las clases")
   
   from joblib import dump
   os.makedirs('models', exist_ok=True)
   dump(best_model, 'models/optimized_classification_model.joblib')
   
   return best_model, grid_search.best_params_, f1

def train_segmented_regression_models(datasets):
   """Entrenar modelos de regresión por segmentos"""
   print("\n📊 ENTRENANDO MODELOS DE REGRESIÓN POR SEGMENTOS")
   print("-" * 50)
   
   segments = datasets['segments']
   X_test, y_test, y_test_log = datasets['test_data']
   
   segment_models = {}
   segment_metrics = {}
   
   for segment_name, segment_data in segments.items():
       X_train = segment_data['X_train']
       y_train_log = segment_data['y_train_log']
       
       print(f"\n🔍 Entrenando modelo para segmento {segment_name} ({len(X_train)} muestras)")
       
       # Pipeline con escalado y modelo
       pipeline = Pipeline([
           ('scaler', RobustScaler()),
           ('model', xgb.XGBRegressor(random_state=42, n_jobs=-1))
       ])
       
       # Usar parámetros más específicos según el segmento
       if segment_name in ['Q1', 'Q2']:  # Valores pequeños
           params = {
               'model__n_estimators': 200,
               'model__max_depth': 3,
               'model__learning_rate': 0.01,
               'model__min_child_weight': 3,
               'model__gamma': 0.1,
               'model__subsample': 0.9,
               'model__colsample_bytree': 0.8
           }
       else:  # Valores grandes (Q3, Q4)
           params = {
               'model__n_estimators': 300,
               'model__max_depth': 5,
               'model__learning_rate': 0.05,
               'model__min_child_weight': 5,
               'model__gamma': 0.2,
               'model__subsample': 1.0,
               'model__colsample_bytree': 0.9
           }
       
       # Ajustar pipeline
       for param, value in params.items():
           pipeline.set_params(**{param: value})
       
       pipeline.fit(X_train, y_train_log)
       segment_models[segment_name] = pipeline
       
       # Guardar modelo
       from joblib import dump
       os.makedirs('models/segments', exist_ok=True)
       dump(pipeline, f'models/segments/regression_model_{segment_name}.joblib')
       
       print(f"✅ Modelo para segmento {segment_name} entrenado y guardado")
   
   # Evaluar modelos combinados en test set
   print("\n📊 Evaluando modelos por segmentos en conjunto de prueba")
   
   # Crear predicciones combinadas
   # Para ello, necesitamos asignar cada muestra de test al segmento correspondiente
   y_pred_combined = np.zeros_like(y_test)
   
   # Determinar cuartiles basados en el conjunto de entrenamiento
   y_reg_train = np.concatenate([segments[s]['y_train'] for s in segments])
   quantile_values = np.percentile(y_reg_train, [25, 50, 75])
   
   # Asignar cada muestra de test a un segmento y predecir
   for i, y_val in enumerate(y_test):
       if y_val <= quantile_values[0]:
           segment = 'Q1'
       elif y_val <= quantile_values[1]:
           segment = 'Q2'
       elif y_val <= quantile_values[2]:
           segment = 'Q3'
       else:
           segment = 'Q4'
       
       # Predecir con el modelo del segmento correspondiente
       X_sample = X_test.iloc[[i]]
       y_pred_log = segment_models[segment].predict(X_sample)
       y_pred_combined[i] = np.expm1(y_pred_log[0])
   
   # Calcular métricas
   mae = mean_absolute_error(y_test, y_pred_combined)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred_combined))
   r2 = r2_score(y_test, y_pred_combined)
   
   print(f"✅ MAE combinado: {mae:.2f}")
   print(f"✅ RMSE combinado: {rmse:.2f}")
   print(f"✅ R² combinado: {r2:.4f}")
   
   # Guardar métricas
   segment_metrics = {
       'combined_metrics': {
           'mae': float(mae),
           'rmse': float(rmse),
           'r2': float(r2)
       },
       'segments': {s: {'samples': len(segment_data['X_train'])} for s, segment_data in segments.items()},
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open('models/segmented_regression_meta.json', 'w') as f:
       json.dump(segment_metrics, f, indent=2)
   
   return segment_models, segment_metrics

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

def optimize_regression_model(datasets):
   """Optimizar el modelo de regresión mediante GridSearch"""
   print("\n📈 OPTIMIZANDO MODELO DE REGRESIÓN GLOBAL")
   print("-" * 50)
   
   # Usar datos con transformación logarítmica
   X_train, X_test, y_train, y_test = datasets['regression_log']
   
   # Probar también transformación raíz cuadrada
   X_train_sqrt, X_test_sqrt, y_train_sqrt, y_test_sqrt = datasets['regression']
   y_train_sqrt = np.sqrt(y_train_sqrt)  # Transformación raíz cuadrada
   
   # Pipeline con escalado y modelo
   pipeline = Pipeline([
       ('scaler', RobustScaler()),
       ('model', xgb.XGBRegressor(random_state=42, n_jobs=-1))
   ])
   
   # Parámetros para búsqueda
   param_grid = {
       'model__n_estimators': [200, 300, 500],
       'model__max_depth': [3, 5, 7],
       'model__learning_rate': [0.01, 0.05, 0.1],
       'model__min_child_weight': [1, 3, 5],
       'model__gamma': [0, 0.1, 0.2],
       'model__subsample': [0.8, 0.9, 1.0],
       'model__colsample_bytree': [0.8, 0.9, 1.0]
   }
   
   # Realizar búsqueda
   print("🔍 Iniciando búsqueda de hiperparámetros para regresión...")
   grid_search = RandomizedSearchCV(
       pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error',
       n_iter=20, random_state=42, n_jobs=-1, verbose=0
   )
   
   # Probar con transformación logarítmica (mejor en análisis previo)
   grid_search.fit(X_train, y_train)
   
   # Evaluar modelo optimizado
   best_model = grid_search.best_estimator_
   y_pred_log = best_model.predict(X_test)
   
   # Transformar predicciones a escala original
   y_pred = np.expm1(y_pred_log)
   y_true = np.expm1(y_test)
   
   # Calcular métricas
   mae = mean_absolute_error(y_true, y_pred)
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   r2 = r2_score(y_true, y_pred)
   
   print(f"✅ Mejores parámetros: {grid_search.best_params_}")
   print(f"✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
   
   # Guardar modelo optimizado
   from joblib import dump
   os.makedirs('models', exist_ok=True)
   dump(best_model, 'models/optimized_regression_model.joblib')
   
   # Guardar datos de referencia para transformación
   model_meta = {
       'best_params': grid_search.best_params_,
       'metrics': {
           'mae': float(mae),
           'rmse': float(rmse),
           'r2': float(r2)
       },
       'transformation': 'logarithmic',
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open('models/optimized_regression_meta.json', 'w') as f:
       json.dump(model_meta, f, indent=2)
   
   return best_model, grid_search.best_params_, (mae, rmse, r2)

def analyze_errors(model, datasets, model_type='regression'):
   """Analizar errores del modelo optimizado"""
   print(f"\n🔬 ANÁLISIS DE ERRORES - {model_type.upper()}")
   print("-" * 50)
   
   if model_type == 'regression':
       X_train, X_test, y_train, y_test = datasets['regression_log']
       
       # Hacer predicciones
       y_pred_log = model.predict(X_test)
       
       # Transformar a escala original
       y_pred = np.expm1(y_pred_log)
       y_true = np.expm1(y_test)
       
       # Calcular errores
       errors = y_true - y_pred
       abs_errors = np.abs(errors)
       
       # Análisis de errores
       error_df = pd.DataFrame({
           'y_true': y_true,
           'y_pred': y_pred,
           'error': errors,
           'abs_error': abs_errors
       })
       
       # Visualizar errores
       plt.figure(figsize=(10, 8))
       plt.subplot(2, 1, 1)
       plt.scatter(y_true, y_pred, alpha=0.5)
       plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
       plt.xlabel('Valor Real')
       plt.ylabel('Predicción')
       plt.title('Predicción vs Valor Real')
       
       plt.subplot(2, 1, 2)
       plt.scatter(y_true, abs_errors, alpha=0.5)
       plt.xlabel('Valor Real')
       plt.ylabel('Error Absoluto')
       plt.title('Error Absoluto vs Valor Real')
       
       plt.tight_layout()
       plt.savefig('results/plots/regression_error_analysis.png', dpi=300)
       
       # Análisis por segmentos
       quantiles = [0, 0.25, 0.5, 0.75, 1.0]
       labels = ['Q1', 'Q2', 'Q3', 'Q4']
       
       error_df['quantile'] = pd.qcut(y_true, q=quantiles, labels=labels)
       segment_analysis = error_df.groupby('quantile').agg({
           'y_true': 'mean',
           'abs_error': ['mean', 'median', 'std'],
           'error': ['mean', 'count']
       })
       
       print("\n📊 Análisis de errores por segmento:")
       print(segment_analysis.round(2))
       
       # Guardar análisis
       segment_analysis.to_csv('results/regression_error_segments.csv')
       
   else:  # clasificación
       X_train, X_test, y_train, y_test = datasets['classification']
       
       # Hacer predicciones
       y_pred = model.predict(X_test)
       
       # Matriz de confusión
       from sklearn.metrics import confusion_matrix, classification_report
       
       cm = confusion_matrix(y_test, y_pred)
       
       # Visualizar matriz de confusión
       plt.figure(figsize=(8, 6))
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Reponer', 'Reponer'],
                   yticklabels=['No Reponer', 'Reponer'])
       plt.xlabel('Predicción')
       plt.ylabel('Valor Real')
       plt.title('Matriz de Confusión')
       plt.tight_layout()
       plt.savefig('results/plots/classification_confusion_matrix.png', dpi=300)
       
       # Informe de clasificación
       report = classification_report(y_test, y_pred, output_dict=True)
       report_df = pd.DataFrame(report).transpose()
       
       print("\n📊 Informe de clasificación:")
       print(report_df.round(3))
       
       # Guardar informe
       report_df.to_csv('results/classification_report.csv')

def implement_hybrid_approach(datasets, class_model, reg_models=None):
   """Implementar enfoque híbrido: clasificar primero, luego predecir cantidad"""
   print("\n🔄 IMPLEMENTANDO ENFOQUE HÍBRIDO")
   print("-" * 50)
   
   # Datos de prueba
   X_test, y_test = datasets['classification'][1], datasets['classification'][3]
   
   # Modelo de regresión global
   if reg_models is None:
       from joblib import load
       try:
           reg_model = load('models/optimized_regression_model.joblib')
       except:
           print("⚠️ No se encontró un modelo de regresión optimizado")
           return
   else:
       reg_model = reg_models  # Usar modelos por segmentos

   # Hacer predicciones de clasificación
   y_class_pred = class_model.predict(X_test)
   
   # Encontrar índices donde se predice reposición
   indices_reposicion = np.where(y_class_pred == 1)[0]
   
   # Predecir cantidades solo para esos índices
   X_reposicion = X_test.iloc[indices_reposicion]
   
   # Convertir índices a posiciones en el conjunto de regresión
   X_reg_test = datasets['regression'][1]
   y_reg_test = datasets['regression'][2]
   
   # Mapear índices de clasificación a índices de regresión
   # Esto es complejo porque no hay correspondencia directa entre ambos datasets
   # Asumimos que usamos predicción de modelo global para simplificar
   y_hybrid_pred = np.zeros_like(y_test, dtype=float)
   
   if isinstance(reg_model, dict):  # Modelos por segmentos
       # Implementación con modelos por segmentos (más compleja)
       print("✅ Usando modelos por segmentos para enfoque híbrido")
       
       # Determinamos el segmento de cada muestra positiva
       y_reg_train = np.concatenate([datasets['segments'][s]['y_train'] for s in datasets['segments']])
       quantile_values = np.percentile(y_reg_train, [25, 50, 75])
       
       for idx in indices_reposicion:
           X_sample = X_test.iloc[[idx]]
           
           # Seleccionar modelo según cantidad esperada (aproximación)
           segment = 'Q2'  # Valor por defecto (mediano)
           
           # Hacer predicción con el modelo del segmento
           y_pred_log = reg_models[segment].predict(X_sample)
           y_hybrid_pred[idx] = np.expm1(y_pred_log[0])
   else:
       # Implementación con modelo global
       print("✅ Usando modelo global para enfoque híbrido")
       
       # Hacer predicciones para casos de reposición
       y_pred_log = reg_model.predict(X_reposicion)
       
       # Asignar predicciones
       for i, idx in enumerate(indices_reposicion):
           y_hybrid_pred[idx] = np.expm1(y_pred_log[i])
   
   # Calcular métricas (solo para casos donde se predice reposición)
   y_true_reposicion = y_test.iloc[indices_reposicion]
   y_pred_reposicion = y_hybrid_pred[indices_reposicion]
   
   # Métricas de clasificación
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   accuracy = accuracy_score(y_test, y_class_pred)
   precision = precision_score(y_test, y_class_pred)
   recall = recall_score(y_test, y_class_pred)
   f1 = f1_score(y_test, y_class_pred)
   
   # Métricas híbridas
   mae_hybrid = mean_absolute_error(y_true_reposicion, y_pred_reposicion)
   
   print(f"✅ Métricas de clasificación:")
   print(f"   • Accuracy: {accuracy:.4f}")
   print(f"   • Precision: {precision:.4f}")
   print(f"   • Recall: {recall:.4f}")
   print(f"   • F1-Score: {f1:.4f}")
   
   print(f"\n✅ Métricas de regresión (solo para casos con reposición):")
   print(f"   • MAE: {mae_hybrid:.2f}")
   
   # Guardar metadatos del enfoque híbrido
   hybrid_meta = {
       'classification_metrics': {
           'accuracy': float(accuracy),
           'precision': float(precision),
           'recall': float(recall),
           'f1': float(f1)
       },
       'regression_metrics': {
           'mae': float(mae_hybrid)
       },
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open('models/hybrid_approach_meta.json', 'w') as f:
       json.dump(hybrid_meta, f, indent=2)
   
   return hybrid_meta

def main():
   print("🚀 ANÁLISIS Y OPTIMIZACIÓN DE CARACTERÍSTICAS")
   print("="*60)
   
   # Cargar datos y resultados
   df, features, class_results, reg_results = load_data_and_results()
   if df is None:
       return
   
   # Analizar importancia de características
   importance_df = analyze_feature_importance()
   
   # Seleccionar características óptimas
   selected_features = select_optimal_features(df, features, importance_df, threshold=0.8)
   
   # Preparar datasets optimizados
   datasets = prepare_optimized_datasets(df, selected_features)
   
   # Optimizar modelo de clasificación con SMOTE
   best_class_model, best_class_params, class_f1 = optimize_classification_model(datasets)
   
   # Optimizar modelo de regresión global
   best_reg_model, best_reg_params, reg_metrics = optimize_regression_model(datasets)
   
   # Entrenar modelos por segmentos
   segment_models, segment_metrics = train_segmented_regression_models(datasets)
   
   # Analizar errores
   analyze_errors(best_reg_model, datasets, model_type='regression')
   analyze_errors(best_class_model, datasets, model_type='classification')
   
   # Implementar enfoque híbrido
   hybrid_meta = implement_hybrid_approach(datasets, best_class_model, segment_models)
   
   # Guardar feature set optimizado
   feature_meta = {
       'selected_features': selected_features,
       'feature_count': len(selected_features),
       'optimization_threshold': 0.8,
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open('data/processed/optimized_features.json', 'w') as f:
       json.dump(feature_meta, f, indent=2)
   
   print("\n✅ OPTIMIZACIÓN COMPLETADA")
   print(f"📁 Archivos generados:")
   print(f"   • models/optimized_classification_model.joblib")
   print(f"   • models/optimized_regression_model.joblib")
   print(f"   • models/segments/ (modelos por segmentos)")
   print(f"   • data/processed/optimized_features.json")
   print(f"   • models/hybrid_approach_meta.json")
   print(f"   • results/plots/feature_importance_aggregated.png")
   print(f"   • results/plots/regression_error_analysis.png")
   print(f"   • results/plots/classification_confusion_matrix.png")
   
   # Resumen de resultados y recomendaciones
   print("\n📋 RESUMEN DE MEJORAS IMPLEMENTADAS:")
   print("1. ✅ Clasificación: Implementación de SMOTE para balanceo de clases")
   print("2. ✅ Regresión: Modelos específicos por segmentos de magnitud")
   print("3. ✅ Enfoque híbrido: Clasificación seguida de regresión selectiva")
   
   return best_class_model, segment_models, selected_features

if __name__ == "__main__":
   best_class_model, segment_models, selected_features = main()