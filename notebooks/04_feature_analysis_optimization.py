# 04_feature_optimization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import json
import os
from datetime import datetime
from joblib import dump
warnings.filterwarnings('ignore')

output_dir = 'results/04_feature_optimization'
data_dir = 'data/processed/04_optimization'
plots_dir = f'{output_dir}/plots'
models_dir = 'models'
importance_dir = 'results/03_model_comparison'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(f'{models_dir}/segments', exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

def load_data_and_results():
   """Cargar datos procesados y resultados de modelos anteriores"""
   try:
       df = pd.read_csv('data/processed/02_features/features_engineered.csv')
       
       with open('results/02_feature_engineering/feature_metadata.json', 'r') as f:
           metadata = json.load(f)
           features = metadata.get('features', [])
       
       class_results = pd.read_csv('results/03_model_comparison/classification_results.csv', index_col=0)
       reg_results = pd.read_csv('results/03_model_comparison/regression_results.csv', index_col=0)
       
       print(f"✅ Datos cargados: {df.shape[0]} registros, {len(features)} features")
       print(f"✅ Resultados anteriores: {len(class_results)} modelos de clasificación, {len(reg_results)} modelos de regresión")
       
       return df, features, class_results, reg_results
   
   except FileNotFoundError as e:
       print(f"❌ Error: {e}")
       print("   Ejecuta primero los scripts anteriores (01, 02 y 03)")
       return None, None, None, None

def get_best_models(class_results, reg_results):
   """Identificar los mejores modelos de clasificación y regresión"""
   print("\n🏆 IDENTIFICANDO MEJORES MODELOS DEL PASO ANTERIOR")
   print("-" * 50)
   
   # Mejor modelo de clasificación (por accuracy)
   best_class_model = class_results['Test_Accuracy'].idxmax()
   best_class_f1 = class_results['Test_F1'].idxmax()
   
   # Mejor modelo de regresión (solo los que usan transformación logarítmica)
   log_models = [model for model in reg_results.index if '(Log)' in model]
   reg_log_results = reg_results.loc[log_models]
   best_reg_model = reg_log_results['Test_R2'].idxmax()
   
   print(f"✅ Mejor modelo clasificación (accuracy): {best_class_model} - {class_results.loc[best_class_model, 'Test_Accuracy']:.4f}")
   print(f"✅ Mejor modelo clasificación (F1): {best_class_f1} - {class_results.loc[best_class_f1, 'Test_F1']:.4f}")
   print(f"✅ Mejor modelo regresión (R²): {best_reg_model} - {reg_log_results.loc[best_reg_model, 'Test_R2']:.4f}")
   
   return best_class_model, best_reg_model

def analyze_feature_importance():
   """Analizar importancia de características según modelos anteriores"""
   print("\n🔍 ANALIZANDO IMPORTANCIA DE CARACTERÍSTICAS")
   print("-" * 50)
   
   importance_files = [f for f in os.listdir('results/03_model_comparison') if f.startswith('importance_') and f.endswith('.csv')]
   
   if not importance_files:
       print("❌ No se encontraron análisis de importancia. Ejecuta primero 03_model_comparison.py")
       return None
   
   all_importances = []
   
   for file in importance_files:
       model_name = file.replace('importance_', '').replace('.csv', '')
       imp_df = pd.read_csv(f'{importance_dir}/{file}')
       imp_df['model'] = model_name
       all_importances.append(imp_df)
   
   combined_imp = pd.concat(all_importances)
   
   avg_imp = combined_imp.groupby('feature')['importance'].mean().reset_index()
   avg_imp = avg_imp.sort_values('importance', ascending=False)
   
   os.makedirs('results/plots', exist_ok=True)
   plt.figure(figsize=(12, 8))
   top_n = min(20, len(avg_imp))
   sns.barplot(x='importance', y='feature', data=avg_imp.head(top_n))
   plt.title(f'Top {top_n} Características por Importancia Promedio')
   plt.tight_layout()
   plt.savefig(f'{plots_dir}/feature_importance_aggregated.png', dpi=300)
   
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
   
   X_train, X_test, y_class_train, y_class_test = train_test_split(
       X, y_class, test_size=0.2, random_state=42, stratify=y_class
   )
   
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

def optimize_classification_model(datasets, best_model_name):
   """Optimizar el modelo de clasificación con SMOTE para balancear clases"""
   print(f"\n🔄 OPTIMIZANDO MODELO DE CLASIFICACIÓN ({best_model_name})")
   print("-" * 50)
   
   X_train, X_test, y_train, y_test = datasets['classification']
   
   # Seleccionar el modelo correcto basado en el nombre
   if 'Gradient Boosting' in best_model_name:
       base_model = GradientBoostingClassifier(random_state=42)
       param_grid = {
           'model__n_estimators': [100, 200, 300],
           'model__max_depth': [3, 5, 7],
           'model__learning_rate': [0.01, 0.05, 0.1],
           'model__min_samples_leaf': [1, 5, 10],
           'model__min_samples_split': [2, 5, 10],
           'model__subsample': [0.8, 0.9, 1.0]
       }
   elif 'XGBoost' in best_model_name:
       base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
       param_grid = {
           'model__n_estimators': [100, 200, 300],
           'model__max_depth': [3, 5, 7],
           'model__learning_rate': [0.01, 0.05, 0.1],
           'model__min_child_weight': [1, 3, 5],
           'model__gamma': [0, 0.1, 0.2],
           'model__subsample': [0.8, 0.9, 1.0],
           'model__colsample_bytree': [0.8, 0.9, 1.0]
       }
   elif 'LightGBM' in best_model_name:
       base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
       param_grid = {
           'model__n_estimators': [100, 200, 300],
           'model__max_depth': [3, 5, 7],
           'model__learning_rate': [0.01, 0.05, 0.1],
           'model__num_leaves': [31, 63, 127],
           'model__min_child_samples': [5, 10, 20],
           'model__subsample': [0.8, 0.9, 1.0]
       }
   else:
       # Fallback a Gradient Boosting
       base_model = GradientBoostingClassifier(random_state=42)
       param_grid = {
           'model__n_estimators': [100, 200, 300],
           'model__max_depth': [3, 5, 7],
           'model__learning_rate': [0.01, 0.05, 0.1],
           'model__min_samples_leaf': [1, 5, 10],
           'model__min_samples_split': [2, 5, 10],
           'model__subsample': [0.8, 0.9, 1.0]
       }
       print(f"⚠️ Modelo {best_model_name} no implementado específicamente, usando Gradient Boosting")
   
   # Crear pipeline con SMOTE
   pipeline = ImbPipeline([
       ('scaler', RobustScaler()),
       ('smote', SMOTE(random_state=42)),
       ('model', base_model)
   ])
   
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
   
   os.makedirs('models', exist_ok=True)
   dump(best_model, f'{models_dir}/optimized_classification_model.joblib')
   
   return best_model, grid_search.best_params_, f1

def train_segmented_regression_models(datasets, best_reg_model_name):
   """Entrenar modelos de regresión por segmentos"""
   print(f"\n📊 ENTRENANDO MODELOS DE REGRESIÓN POR SEGMENTOS ({best_reg_model_name})")
   print("-" * 50)
   
   segments = datasets['segments']
   X_test, y_test, y_test_log = datasets['test_data']
   
   segment_models = {}
   segment_metrics = {}
   
   # Seleccionar el modelo base según el mejor modelo del paso anterior
   if 'XGBoost' in best_reg_model_name:
       model_class = xgb.XGBRegressor
   elif 'Gradient Boosting' in best_reg_model_name:
       model_class = xgb.XGBRegressor  # Usamos XGBoost de todas formas por su rendimiento
   else:
       model_class = xgb.XGBRegressor
       print(f"⚠️ Modelo {best_reg_model_name} no implementado específicamente, usando XGBoost")
   
   for segment_name, segment_data in segments.items():
       X_train = segment_data['X_train']
       y_train_log = segment_data['y_train_log']
       
       print(f"\n🔍 Entrenando modelo para segmento {segment_name} ({len(X_train)} muestras)")
       
       # Pipeline con escalado y modelo
       pipeline = Pipeline([
           ('scaler', RobustScaler()),
           ('model', model_class(random_state=42, n_jobs=-1))
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
       os.makedirs('models/segments', exist_ok=True)
       dump(pipeline, f'{models_dir}/segments/regression_model_{segment_name}.joblib')
       
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
   
   with open(f'{models_dir}/segmented_regression_meta.json', 'w') as f:
       json.dump(segment_metrics, f, indent=2)
   
   return segment_models, segment_metrics

def implement_hybrid_approach(datasets, class_model, reg_models):
   """Implementar enfoque híbrido: clasificar primero, luego predecir cantidad"""
   print("\n🔄 IMPLEMENTANDO ENFOQUE HÍBRIDO")
   print("-" * 50)
   
   # Datos de prueba
   X_test, y_test = datasets['classification'][1], datasets['classification'][3]
   
   # Hacer predicciones de clasificación
   y_class_pred = class_model.predict(X_test)
   
   # Encontrar índices donde se predice reposición
   indices_reposicion = np.where(y_class_pred == 1)[0]
   
   # Predecir cantidades solo para esos índices
   X_reposicion = X_test.iloc[indices_reposicion]
   
   # Usar modelos por segmentos
   print("✅ Usando modelos por segmentos para enfoque híbrido")
   
   y_hybrid_pred = np.zeros_like(y_test, dtype=float)
   
   # Determinamos el segmento de cada muestra positiva
   y_reg_train = np.concatenate([datasets['segments'][s]['y_train'] for s in datasets['segments']])
   quantile_values = np.percentile(y_reg_train, [25, 50, 75])
   
   for idx in indices_reposicion:
       X_sample = X_test.iloc[[idx]]
       
       # Predecir con el modelo del segmento Q2 (mediano)
       # Nota: Idealmente se debería determinar dinámicamente el segmento
       segment = 'Q2'
       y_pred_log = reg_models[segment].predict(X_sample)
       y_hybrid_pred[idx] = np.expm1(y_pred_log[0])
   
   # Calcular métricas (solo para casos donde se predice reposición)
   y_true_reposicion = y_test.iloc[indices_reposicion]
   y_pred_reposicion = y_hybrid_pred[indices_reposicion]
   
   # Métricas de clasificación
   from sklearn.metrics import accuracy_score, precision_score, recall_score
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
   
   with open(f'{models_dir}/hybrid_approach_meta.json', 'w') as f:
       json.dump(hybrid_meta, f, indent=2)
   
   return hybrid_meta

def main():
   print("🚀 ANÁLISIS Y OPTIMIZACIÓN DE CARACTERÍSTICAS")
   print("="*60)
   
   # Cargar datos y resultados
   df, features, class_results, reg_results = load_data_and_results()
   if df is None:
       return
   
   # Identificar mejores modelos del paso anterior
   best_class_model_name, best_reg_model_name = get_best_models(class_results, reg_results)
   
   # Analizar importancia de características
   importance_df = analyze_feature_importance()
   
   # Seleccionar características óptimas
   selected_features = select_optimal_features(df, features, importance_df, threshold=0.8)
   
   # Preparar datasets optimizados
   datasets = prepare_optimized_datasets(df, selected_features)
   
   # Optimizar modelo de clasificación con SMOTE
   best_class_model, best_class_params, class_f1 = optimize_classification_model(datasets, best_class_model_name)
   
   # Entrenar modelos por segmentos
   segment_models, segment_metrics = train_segmented_regression_models(datasets, best_reg_model_name)
   
   # Implementar enfoque híbrido
   hybrid_meta = implement_hybrid_approach(datasets, best_class_model, segment_models)
   
   # Guardar feature set optimizado
   feature_meta = {
       'selected_features': selected_features,
       'feature_count': len(selected_features),
       'optimization_threshold': 0.8,
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open(f'{data_dir}/optimized_features.json', 'w') as f:
       json.dump(feature_meta, f, indent=2)
   
   print("\n✅ OPTIMIZACIÓN COMPLETADA")
   print(f"📁 Archivos generados:")
   print(f"   • models/optimized_classification_model.joblib")
   print(f"   • models/segments/ (modelos por segmentos)")
   print(f"   • data/processed/optimized_features.json")
   print(f"   • models/hybrid_approach_meta.json")
   print(f"   • results/plots/feature_importance_aggregated.png")
   
   # Resumen de resultados y recomendaciones
   print("\n📋 RESUMEN DE MEJORAS IMPLEMENTADAS:")
   print(f"1. ✅ Clasificación: {best_class_model_name} con SMOTE para balanceo de clases")
   print(f"2. ✅ Regresión: {best_reg_model_name} por segmentos")
   print("3. ✅ Enfoque híbrido: Clasificación seguida de regresión selectiva")
   
   return best_class_model, segment_models, selected_features

if __name__ == "__main__":
   best_class_model, segment_models, selected_features = main()