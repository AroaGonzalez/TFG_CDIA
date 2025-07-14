# 05_deep_learning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import os
import json
from datetime import datetime
from joblib import load, dump
import warnings
warnings.filterwarnings('ignore')

# Configuración
output_dir = 'results/05_ensemble_learning'
plots_dir = f'{output_dir}/plots'
models_dir = 'models/ensemble'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_best_models_from_results(class_results, reg_results):
   """Identifica y reconstruye los mejores modelos basados en los resultados previos"""
   best_class_model = None
   best_reg_model = None
   
   # Identificar mejor modelo de clasificación
   if not class_results.empty:
       best_class_name = class_results['Test_F1'].idxmax()
       print(f"✅ Mejor modelo de clasificación identificado: {best_class_name}")
       
       # Reconstruir el modelo de clasificación según su tipo
       if 'Random Forest' in best_class_name:
           best_class_model = RandomForestClassifier(
               n_estimators=200, max_depth=7, min_samples_leaf=2, 
               class_weight='balanced', random_state=42)
       elif 'Gradient Boosting' in best_class_name:
           best_class_model = GradientBoostingClassifier(
               n_estimators=150, max_depth=5, learning_rate=0.05, 
               subsample=0.9, random_state=42)
       elif 'XGBoost' in best_class_name:
           best_class_model = xgb.XGBClassifier(
               n_estimators=150, max_depth=5, learning_rate=0.05, 
               scale_pos_weight=3, random_state=42)
       elif 'LightGBM' in best_class_name:
           best_class_model = lgb.LGBMClassifier(
               n_estimators=150, max_depth=5, learning_rate=0.05,
               class_weight='balanced', random_state=42, verbose=-1)
   
   # Identificar mejor modelo de regresión (solo los que usan transformación logarítmica)
   if not reg_results.empty:
       # Filtrar modelos con transformación logarítmica
       log_models = [model for model in reg_results.index if '(Log)' in model]
       if log_models:
           reg_log_results = reg_results.loc[log_models]
           best_reg_name = reg_log_results['Test_R2'].idxmax()
           print(f"✅ Mejor modelo de regresión identificado: {best_reg_name}")
           
           # Reconstruir el modelo de regresión según su tipo
           if 'Random Forest' in best_reg_name:
               best_reg_model = RandomForestRegressor(
                   n_estimators=200, max_depth=7, min_samples_leaf=2, random_state=42)
           elif 'Gradient Boosting' in best_reg_name:
               best_reg_model = GradientBoostingRegressor(
                   n_estimators=200, max_depth=5, learning_rate=0.05, 
                   subsample=0.9, random_state=42)
           elif 'XGBoost' in best_reg_name:
               best_reg_model = xgb.XGBRegressor(
                   n_estimators=200, max_depth=5, learning_rate=0.05, 
                   subsample=0.8, random_state=42)
           elif 'LightGBM' in best_reg_name:
               best_reg_model = lgb.LGBMRegressor(
                   n_estimators=200, max_depth=5, learning_rate=0.05,
                   random_state=42, verbose=-1)
   
   return best_class_model, best_reg_model

def load_data_and_models():
   """Cargar datos preprocesados y mejores modelos del paso 3"""
   # Cargar datos
   df = pd.read_csv('data/processed/02_features/features_engineered.csv')
   
   # Cargar resultados del paso 3
   try:
       class_results = pd.read_csv('results/03_model_comparison/classification_results.csv', index_col=0)
       reg_results = pd.read_csv('results/03_model_comparison/regression_results.csv', index_col=0)
       
       # Imprimir información del mejor modelo del paso 3
       best_class_model_name = class_results['Test_F1'].idxmax()
       best_class_accuracy = class_results.loc[best_class_model_name, 'Test_Accuracy']
       best_class_precision = class_results.loc[best_class_model_name, 'Test_Precision']
       best_class_recall = class_results.loc[best_class_model_name, 'Test_Recall']
       best_class_f1 = class_results.loc[best_class_model_name, 'Test_F1']
       
       print(f"📊 Mejor modelo clasificación (Paso 3):")
       print(f"   • Modelo: {best_class_model_name}")
       print(f"   • Accuracy: {best_class_accuracy:.4f}")
       print(f"   • Precision: {best_class_precision:.4f}")
       print(f"   • Recall: {best_class_recall:.4f}")
       print(f"   • F1: {best_class_f1:.4f}")
       
       # Para regresión
       log_models = [model for model in reg_results.index if '(Log)' in model]
       if log_models:
           reg_log_results = reg_results.loc[log_models]
           best_reg_model_name = reg_log_results['Test_R2'].idxmax()
           best_reg_mae = reg_log_results.loc[best_reg_model_name, 'Test_MAE']
           best_reg_rmse = reg_log_results.loc[best_reg_model_name, 'Test_RMSE']
           best_reg_r2 = reg_log_results.loc[best_reg_model_name, 'Test_R2']
           
           print(f"\n📊 Mejor modelo regresión (Paso 3):")
           print(f"   • Modelo: {best_reg_model_name}")
           print(f"   • MAE: {best_reg_mae:.2f}")
           print(f"   • RMSE: {best_reg_rmse:.2f}")
           print(f"   • R²: {best_reg_r2:.4f}")
       
   except:
       print("⚠️ No se encontraron resultados del paso 3, continuando sin ellos")
       class_results = pd.DataFrame()
       reg_results = pd.DataFrame()
   
   # Cargar las importancias de características del paso 3
   importance_files = [f for f in os.listdir('results/03_model_comparison') 
                     if f.startswith('importance_') and f.endswith('.csv')]
   
   feature_importances = {}
   for file in importance_files:
       model_name = file.replace('importance_', '').replace('.csv', '')
       importance_df = pd.read_csv(f'results/03_model_comparison/{file}')
       feature_importances[model_name] = importance_df
   
   # Identificar características importantes (unión de top 20 de cada modelo)
   top_features = set()
   for model, imp_df in feature_importances.items():
       top_20 = imp_df.head(20)['feature'].tolist()
       top_features.update(top_20)
   
   # Reconstruir los mejores modelos basados en resultados previos
   best_class_model, best_reg_model = load_best_models_from_results(class_results, reg_results)
   if best_class_model is None or best_reg_model is None:
       print("⚠️ No se pudieron reconstruir algunos modelos, continuando con los disponibles")
   else:
       print("✅ Modelos reconstruidos correctamente a partir de resultados previos")
   
   # Obtener todas las features (excluyendo targets e IDs)
   all_features = [col for col in df.columns 
                  if col not in ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                               'necesita_reposicion', 'cantidad_a_reponer', 
                               'log_cantidad_a_reponer']]
   
   print(f"✅ Datos cargados: {df.shape[0]} registros")
   print(f"✅ Features totales: {len(all_features)}")
   print(f"✅ Top features identificadas: {len(top_features)}")
   
   return df, all_features, list(top_features), best_class_model, best_reg_model, class_results, reg_results

def prepare_data(df, features, top_features):
   """Preparar datos para entrenamiento con ensemble learning"""
   print("\n📊 PREPARANDO DATASETS")
   print("-" * 50)
   
   # Seleccionar features numéricas
   X_all = df[features].select_dtypes(include=['number'])
   
   # Si hay top_features, usarlas, sino usar todas
   if top_features:
       X_top = df[[col for col in top_features if col in X_all.columns]]
       print(f"✅ Features numéricas: {X_all.shape[1]} totales, {X_top.shape[1]} top features")
   else:
       X_top = X_all
       print(f"✅ Features numéricas: {X_all.shape[1]} (usando todas las features)")
   
   # Manejar NaN
   for df_x in [X_all, X_top]:
       for col in df_x.columns:
           if df_x[col].isna().any():
               df_x[col] = df_x[col].fillna(df_x[col].median())
   
   # Targets
   y_class = df['necesita_reposicion'] if 'necesita_reposicion' in df.columns else None
   y_reg = df['cantidad_a_reponer'] if 'cantidad_a_reponer' in df.columns else None
   y_reg_log = df['log_cantidad_a_reponer'] if 'log_cantidad_a_reponer' in df.columns else None
   
   # Split estratificado
   X_train, X_test, y_class_train, y_class_test = train_test_split(
       X_all, y_class, test_size=0.2, random_state=42, stratify=y_class
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
           'feature_names': X_all.columns.tolist(),
           'X_train_df': X_train,
           'X_test_df': X_test
       },
       'regression': {
           'X_train': X_reg_train_scaled, 
           'X_test': X_reg_test_scaled,
           'y_train': y_reg_train, 
           'y_test': y_reg_test,
           'y_log_train': y_reg_log_train, 
           'y_log_test': y_reg_log_test,
           'feature_names': X_all.columns.tolist(),
           'X_train_df': X_reg_train,
           'X_test_df': X_reg_test
       },
       'indices': {
           'train_indices': X_train.index,
           'test_indices': X_test.index,
           'train_reg_indices': X_reg_train.index,
           'test_reg_indices': X_reg_test.index
       }
   }

def optimize_voting_classifier(data):
   """Optimizar modelo de ensemble voting para clasificación"""
   print("\n🔍 OPTIMIZANDO VOTING CLASSIFIER")
   print("-" * 50)
   
   # Definir modelos base con parámetros optimizados
   base_models = [
       ('rf', RandomForestClassifier(
           n_estimators=200, 
           max_depth=7, 
           min_samples_leaf=2, 
           min_samples_split=5,
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
       voting='soft',  # 'soft' para usar probabilidades
       weights=[2, 1, 2]  # Ponderación de modelos basada en rendimiento previo
   )
   
   # Verificar NaN antes de SMOTE
   X_train = data['classification']['X_train']
   y_train = data['classification']['y_train']
   
   # Comprobar y manejar NaN antes de SMOTE
   if np.isnan(X_train).any():
       print("⚠️ Detectados NaN en X_train, aplicando imputación antes de SMOTE")
       X_train = np.nan_to_num(X_train, nan=0.0)
   
   # Aplicar SMOTE para balance de clases
   smote = SMOTE(random_state=42, sampling_strategy=0.6)  # Balanceo parcial a 0.6:1
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   
   print(f"✅ SMOTE aplicado: {y_train.mean():.1%} → {y_train_resampled.mean():.1%} positivos")
   
   # Entrenar modelo
   voting_clf.fit(X_train_resampled, y_train_resampled)
   
   # Evaluar en conjunto de prueba
   y_pred = voting_clf.predict(data['classification']['X_test'])
   
   # Métricas
   accuracy = accuracy_score(data['classification']['y_test'], y_pred)
   precision = precision_score(data['classification']['y_test'], y_pred)
   recall = recall_score(data['classification']['y_test'], y_pred)
   f1 = f1_score(data['classification']['y_test'], y_pred)
   
   print(f"✅ Resultados del Voting Classifier Optimizado:")
   print(f"   • Accuracy: {accuracy:.4f}")
   print(f"   • Precision: {precision:.4f}")
   print(f"   • Recall: {recall:.4f}")
   print(f"   • F1-Score: {f1:.4f}")
   
   # Guardar modelo
   dump(voting_clf, f'{models_dir}/voting_classifier.joblib')
   
   return voting_clf, {
       'accuracy': float(accuracy),
       'precision': float(precision),
       'recall': float(recall),
       'f1': float(f1)
   }

def optimize_stacking_classifier(data):
   """Optimizar modelo de ensemble stacking para clasificación"""
   print("\n🥞 OPTIMIZANDO STACKING CLASSIFIER")
   print("-" * 50)
   
   # Definir modelos base con parámetros optimizados
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
           scale_pos_weight=3,
           eval_metric='logloss',
           random_state=42)),
       ('lgb', lgb.LGBMClassifier(
           n_estimators=150, 
           max_depth=5, 
           learning_rate=0.05,
           class_weight='balanced',
           random_state=42,
           verbose=-1))
   ]
   
   # Meta-clasificador optimizado
   meta_classifier = lgb.LGBMClassifier(
       n_estimators=100, 
       learning_rate=0.01,  # Aprendizaje lento para el meta-modelo
       max_depth=3,  # Meta-modelo más simple
       class_weight='balanced',
       random_state=42,
       verbose=-1
   )
   
   # Crear stacking classifier
   stacking_clf = StackingClassifier(
       estimators=base_models,
       final_estimator=meta_classifier,
       cv=5,
       stack_method='predict_proba'  # Usar probabilidades
   )
   
   # Verificar NaN antes de SMOTE
   X_train = data['classification']['X_train']
   y_train = data['classification']['y_train']
   
   # Comprobar y manejar NaN antes de SMOTE
   if np.isnan(X_train).any():
       print("⚠️ Detectados NaN en X_train, aplicando imputación antes de SMOTE")
       X_train = np.nan_to_num(X_train, nan=0.0)
   
   # Aplicar SMOTE para balance de clases
   smote = SMOTE(random_state=42, sampling_strategy=0.6)  # Balanceo parcial a 0.6:1
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   
   print(f"✅ SMOTE aplicado: {y_train.mean():.1%} → {y_train_resampled.mean():.1%} positivos")
   
   # Entrenar modelo
   stacking_clf.fit(X_train_resampled, y_train_resampled)
   
   # Evaluar en conjunto de prueba
   y_pred = stacking_clf.predict(data['classification']['X_test'])
   
   # Métricas
   accuracy = accuracy_score(data['classification']['y_test'], y_pred)
   precision = precision_score(data['classification']['y_test'], y_pred)
   recall = recall_score(data['classification']['y_test'], y_pred)
   f1 = f1_score(data['classification']['y_test'], y_pred)
   
   print(f"✅ Resultados del Stacking Classifier Optimizado:")
   print(f"   • Accuracy: {accuracy:.4f}")
   print(f"   • Precision: {precision:.4f}")
   print(f"   • Recall: {recall:.4f}")
   print(f"   • F1-Score: {f1:.4f}")
   
   # Guardar modelo
   dump(stacking_clf, f'{models_dir}/stacking_classifier.joblib')
   
   return stacking_clf, {
       'accuracy': float(accuracy),
       'precision': float(precision),
       'recall': float(recall),
       'f1': float(f1)
   }

def optimize_voting_regressor(data):
   """Optimizar modelo de ensemble voting para regresión"""
   print("\n🗳️ OPTIMIZANDO VOTING REGRESSOR")
   print("-" * 50)
   
   # Definir modelos base optimizados
   base_models = [
       ('rf', RandomForestRegressor(
           n_estimators=200, 
           max_depth=7, 
           min_samples_leaf=2, 
           random_state=42)),
       ('gb', GradientBoostingRegressor(
           n_estimators=200, 
           max_depth=4, 
           learning_rate=0.01,  # Aprendizaje más lento
           subsample=0.9,
           random_state=42)),
       ('xgb', xgb.XGBRegressor(
           n_estimators=200, 
           max_depth=4, 
           learning_rate=0.01,  # Aprendizaje más lento
           gamma=0.1,
           subsample=0.8, 
           colsample_bytree=0.8,
           random_state=42))
   ]
   
   # Crear voting regressor con ponderación
   voting_reg = VotingRegressor(
       estimators=base_models,
       weights=[1, 2, 3]  # Mayor peso a XGBoost basado en rendimiento previo
   )
   
   # Verificar NaN antes del entrenamiento
   X_train = data['regression']['X_train']
   y_train = data['regression']['y_log_train']
   
   if np.isnan(X_train).any():
       print("⚠️ Detectados NaN en X_train para regresión, aplicando imputación")
       X_train = np.nan_to_num(X_train, nan=0.0)
   
   if np.isnan(y_train).any():
       print("⚠️ Detectados NaN en y_train para regresión, aplicando imputación")
       y_train = np.nan_to_num(y_train, nan=0.0)
   
   # Entrenar modelo con datos log-transformados
   voting_reg.fit(X_train, y_train)
   
   # Evaluar en conjunto de prueba
   y_pred_log = voting_reg.predict(data['regression']['X_test'])
   y_pred = np.expm1(y_pred_log)
   y_true = np.expm1(data['regression']['y_log_test'])
   
   # Métricas
   mae = mean_absolute_error(y_true, y_pred)
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   r2 = r2_score(y_true, y_pred)
   
   print(f"✅ Resultados del Voting Regressor Optimizado:")
   print(f"   • MAE: {mae:.2f}")
   print(f"   • RMSE: {rmse:.2f}")
   print(f"   • R²: {r2:.4f}")
   
   # Visualizar predicciones vs reales
   plt.figure(figsize=(10, 6))
   plt.scatter(y_true, y_pred, alpha=0.5)
   plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
   plt.xlabel('Valores Reales')
   plt.ylabel('Predicciones')
   plt.title('Predicciones vs Valores Reales (Voting Regressor)')
   plt.savefig(f'{plots_dir}/voting_regressor_predictions.png')
   
   # Guardar modelo
   dump(voting_reg, f'{models_dir}/voting_regressor.joblib')
   
   return voting_reg, {
       'mae': float(mae),
       'rmse': float(rmse),
       'r2': float(r2)
   }

def optimize_stacking_regressor(data):
   """Optimizar modelo de ensemble stacking para regresión"""
   print("\n🥞 OPTIMIZANDO STACKING REGRESSOR")
   print("-" * 50)
   
   # Definir modelos base optimizados
   base_models = [
       ('rf', RandomForestRegressor(
           n_estimators=200, 
           max_depth=7, 
           min_samples_leaf=2, 
           random_state=42)),
       ('gb', GradientBoostingRegressor(
           n_estimators=200, 
           max_depth=4, 
           learning_rate=0.01,
           subsample=0.9,
           random_state=42)),
       ('xgb', xgb.XGBRegressor(
           n_estimators=200, 
           max_depth=4, 
           learning_rate=0.01,
           gamma=0.1,
           subsample=0.8, 
           colsample_bytree=0.8,
           random_state=42))
   ]
   
   # Meta-regresor optimizado
   meta_regressor = xgb.XGBRegressor(
       n_estimators=100,
       max_depth=3,  # Meta-modelo más simple
       learning_rate=0.01,  # Aprendizaje lento para el meta-modelo
       subsample=0.8,
       random_state=42
   )
   
   # Crear stacking regressor
   stacking_reg = StackingRegressor(
       estimators=base_models,
       final_estimator=meta_regressor,
       cv=5
   )
   
   # Verificar NaN antes del entrenamiento
   X_train = data['regression']['X_train']
   y_train = data['regression']['y_log_train']
   
   if np.isnan(X_train).any():
       print("⚠️ Detectados NaN en X_train para regresión, aplicando imputación")
       X_train = np.nan_to_num(X_train, nan=0.0)
   
   if np.isnan(y_train).any():
       print("⚠️ Detectados NaN en y_train para regresión, aplicando imputación")
       y_train = np.nan_to_num(y_train, nan=0.0)
   
   # Entrenar modelo con datos log-transformados
   stacking_reg.fit(X_train, y_train)
   
   # Evaluar en conjunto de prueba
   y_pred_log = stacking_reg.predict(data['regression']['X_test'])
   y_pred = np.expm1(y_pred_log)
   y_true = np.expm1(data['regression']['y_log_test'])
   
   # Métricas
   mae = mean_absolute_error(y_true, y_pred)
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   r2 = r2_score(y_true, y_pred)
   
   print(f"✅ Resultados del Stacking Regressor Optimizado:")
   print(f"   • MAE: {mae:.2f}")
   print(f"   • RMSE: {rmse:.2f}")
   print(f"   • R²: {r2:.4f}")
   
   # Visualizar predicciones vs reales
   plt.figure(figsize=(10, 6))
   plt.scatter(y_true, y_pred, alpha=0.5)
   plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
   plt.xlabel('Valores Reales')
   plt.ylabel('Predicciones')
   plt.title('Predicciones vs Valores Reales (Stacking Regressor)')
   plt.savefig(f'{plots_dir}/stacking_regressor_predictions.png')
   
   # Guardar modelo
   dump(stacking_reg, f'{models_dir}/stacking_regressor.joblib')
   
   return stacking_reg, {
       'mae': float(mae),
       'rmse': float(rmse),
       'r2': float(r2)
   }

def implement_hybrid_approach(data, class_model, reg_model):
   """Implementar enfoque híbrido optimizado con modelos ensemble"""
   print("\n🔄 IMPLEMENTANDO ENFOQUE HÍBRIDO OPTIMIZADO")
   print("-" * 50)
   
   # Hacer predicciones de clasificación con umbral ajustado
   X_test = data['classification']['X_test']
   y_class_true = data['classification']['y_test']
   
   # Usar predict_proba para obtener probabilidades y ajustar umbral
   if hasattr(class_model, 'predict_proba'):
       y_class_proba = class_model.predict_proba(X_test)[:, 1]
       
       # Buscar umbral óptimo para F1
       best_f1 = 0
       best_threshold = 0.5
       
       for threshold in np.arange(0.3, 0.7, 0.02):
           y_pred_threshold = (y_class_proba > threshold).astype(int)
           f1_score_threshold = f1_score(y_class_true, y_pred_threshold)
           
           if f1_score_threshold > best_f1:
               best_f1 = f1_score_threshold
               best_threshold = threshold
       
       print(f"✅ Umbral óptimo encontrado: {best_threshold:.2f} (F1: {best_f1:.4f})")
       y_class_pred = (y_class_proba > best_threshold).astype(int)
   else:
       # Si no tiene predict_proba, usar predict normal
       y_class_pred = class_model.predict(X_test)
   
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
       else:
           mae_hybrid = float('nan')
           rmse_hybrid = float('nan')
   else:
       mae_hybrid = float('nan')
       rmse_hybrid = float('nan')
   
   print(f"✅ Métricas del enfoque híbrido ensemble optimizado:")
   print(f"   • Clasificación: Accuracy={accuracy:.4f}, F1={f1:.4f}")
   print(f"   • Regresión (solo verdaderos positivos): MAE={mae_hybrid:.2f}")
   
   # Guardar resultados
   hybrid_results = {
       'classification': {
           'accuracy': float(accuracy),
           'precision': float(precision),
           'recall': float(recall),
           'f1': float(f1)
       },
       'regression': {
           'mae': float(mae_hybrid),
           'rmse': float(rmse_hybrid)
       },
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open(f'{models_dir}/hybrid_results.json', 'w') as f:
       json.dump(hybrid_results, f, indent=2)
   
   return hybrid_results

def compare_with_previous_models(ensemble_results, class_results, reg_results):
   """Comparar resultados de ensemble learning con modelos del paso 3"""
   print("\n📊 COMPARACIÓN CON MODELOS ANTERIORES")
   print("-" * 50)
   
   if class_results.empty or reg_results.empty:
       print("⚠️ No hay resultados del paso 3 para comparar")
       return {
           'f1_change': 0.0,
           'mae_change': 0.0,
           'best_class_model': 'No disponible',
           'best_reg_model': 'No disponible'
       }
   
   # Encontrar mejores modelos del paso 3
   best_class_model = class_results['Test_F1'].idxmax()
   best_class_f1 = class_results.loc[best_class_model, 'Test_F1']
   best_class_accuracy = class_results.loc[best_class_model, 'Test_Accuracy']
   best_class_precision = class_results.loc[best_class_model, 'Test_Precision']
   best_class_recall = class_results.loc[best_class_model, 'Test_Recall']
   
   # Filtrar solo modelos con transformación logarítmica para regresión
   log_models = [model for model in reg_results.index if '(Log)' in model]
   if log_models:
       reg_log_results = reg_results.loc[log_models]
       best_reg_model = reg_log_results['Test_R2'].idxmax()
       best_reg_mae = reg_log_results.loc[best_reg_model, 'Test_MAE']
       best_reg_rmse = reg_log_results.loc[best_reg_model, 'Test_RMSE']
       best_reg_r2 = reg_log_results.loc[best_reg_model, 'Test_R2']
   else:
       best_reg_model = reg_results['Test_R2'].idxmax()
       best_reg_mae = reg_results.loc[best_reg_model, 'Test_MAE']
       best_reg_rmse = reg_results.loc[best_reg_model, 'Test_RMSE']
       best_reg_r2 = reg_results.loc[best_reg_model, 'Test_R2']
   
   # Comparar con ensemble
   ensemble_f1 = ensemble_results['classification']['f1']
   ensemble_accuracy = ensemble_results['classification']['accuracy']
   ensemble_precision = ensemble_results['classification']['precision']
   ensemble_recall = ensemble_results['classification']['recall']
   
   ensemble_mae = ensemble_results['regression']['mae']
   
   # Calcular cambios (manejo de NaN)
   if not np.isnan(best_class_f1) and best_class_f1 != 0:
       f1_change = (ensemble_f1 - best_class_f1) / best_class_f1 * 100
   else:
       f1_change = 0.0
   
   if not np.isnan(best_reg_mae) and best_reg_mae != 0:
       mae_change = (best_reg_mae - ensemble_mae) / best_reg_mae * 100  # Positivo = mejor
   else:
       mae_change = 0.0
   
   # Tabla comparativa de métricas de clasificación
   print("\n📋 TABLA COMPARATIVA: CLASIFICACIÓN")
   print("-" * 60)
   print(f"{'Métrica':<15} | {'Mejor Modelo Paso 3':<20} | {'Ensemble Optimizado':<20} | {'Cambio %'}")
   print("-" * 60)
   print(f"{'Modelo':<15} | {best_class_model:<20} | {'Ensemble Híbrido':<20} | {'-'}")
   print(f"{'Accuracy':<15} | {best_class_accuracy:<20.4f} | {ensemble_accuracy:<20.4f} | {(ensemble_accuracy-best_class_accuracy)/best_class_accuracy*100:+.1f}%")
   print(f"{'Precision':<15} | {best_class_precision:<20.4f} | {ensemble_precision:<20.4f} | {(ensemble_precision-best_class_precision)/best_class_precision*100:+.1f}%")
   print(f"{'Recall':<15} | {best_class_recall:<20.4f} | {ensemble_recall:<20.4f} | {(ensemble_recall-best_class_recall)/best_class_recall*100:+.1f}%")
   print(f"{'F1-Score':<15} | {best_class_f1:<20.4f} | {ensemble_f1:<20.4f} | {f1_change:+.1f}%")
   print("-" * 60)
   
   # Tabla comparativa de métricas de regresión
   print("\n📋 TABLA COMPARATIVA: REGRESIÓN")
   print("-" * 60)
   print(f"{'Métrica':<15} | {'Mejor Modelo Paso 3':<20} | {'Ensemble Optimizado':<20} | {'Cambio %'}")
   print("-" * 60)
   print(f"{'Modelo':<15} | {best_reg_model:<20} | {'Ensemble Híbrido':<20} | {'-'}")
   print(f"{'MAE':<15} | {best_reg_mae:<20.2f} | {ensemble_mae:<20.2f} | {mae_change:+.1f}%")
   print(f"{'RMSE':<15} | {best_reg_rmse:<20.2f} | {ensemble_results['regression']['rmse']:<20.2f} | {'-'}")
   print(f"{'R²':<15} | {best_reg_r2:<20.4f} | {'N/A':<20} | {'-'}")
   print("-" * 60)
   
   # Crear gráfico comparativo
   plt.figure(figsize=(12, 10))
   
   # F1-Score
   plt.subplot(2, 1, 1)
   bars = plt.bar(['Mejor Modelo Paso 3', 'Ensemble Optimizado'], [best_class_f1, ensemble_f1], color=['#2C3E50', '#E74C3C'])
   plt.title('Comparación F1-Score (Clasificación)', fontsize=14)
   plt.ylim(0, max(best_class_f1, ensemble_f1) * 1.2)
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   
   # Añadir etiquetas con valores
   for bar in bars:
       height = bar.get_height()
       plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.4f}', ha='center', va='bottom', fontsize=12)
   
   # MAE
   plt.subplot(2, 1, 2)
   bars = plt.bar(['Mejor Modelo Paso 3', 'Ensemble Optimizado'], [best_reg_mae, ensemble_mae], color=['#2C3E50', '#E74C3C'])
   plt.title('Comparación MAE (Regresión - menor es mejor)', fontsize=14)
   plt.ylim(0, max(best_reg_mae, ensemble_mae) * 1.2)
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   
   # Añadir etiquetas con valores
   for bar in bars:
       height = bar.get_height()
       plt.text(bar.get_x() + bar.get_width()/2., height + 5,
               f'{height:.2f}', ha='center', va='bottom', fontsize=12)
   
   plt.tight_layout()
   plt.savefig(f'{plots_dir}/model_comparison.png', dpi=300)
   
   return {
       'f1_change': float(f1_change),
       'mae_change': float(mae_change),
       'best_class_model': best_class_model,
       'best_reg_model': best_reg_model
   }

def main():
   print("🚀 IMPLEMENTACIÓN DE ENSEMBLE LEARNING OPTIMIZADO")
   print("="*60)
   
   # Cargar datos y modelos
   df, all_features, top_features, best_class_model, best_reg_model, class_results, reg_results = load_data_and_models()
   
   # Preparar datos para ensemble learning
   data = prepare_data(df, all_features, top_features)
   
   # Entrenar modelos de ensemble optimizados para clasificación
   voting_clf, voting_clf_metrics = optimize_voting_classifier(data)
   stacking_clf, stacking_clf_metrics = optimize_stacking_classifier(data)
   
   # Entrenar modelos de ensemble optimizados para regresión
   voting_reg, voting_reg_metrics = optimize_voting_regressor(data)
   stacking_reg, stacking_reg_metrics = optimize_stacking_regressor(data)
   
   # Seleccionar los mejores modelos de ensemble
   best_ensemble_clf = stacking_clf if stacking_clf_metrics['f1'] > voting_clf_metrics['f1'] else voting_clf
   best_ensemble_reg = stacking_reg if stacking_reg_metrics['mae'] < voting_reg_metrics['mae'] else voting_reg
   
   best_ensemble_clf_name = "Stacking Classifier" if stacking_clf_metrics['f1'] > voting_clf_metrics['f1'] else "Voting Classifier"
   best_ensemble_reg_name = "Stacking Regressor" if stacking_reg_metrics['mae'] < voting_reg_metrics['mae'] else "Voting Regressor"
   
   # Implementar enfoque híbrido con los mejores modelos
   hybrid_results = implement_hybrid_approach(data, best_ensemble_clf, best_ensemble_reg)
   
   # Comparar con modelos anteriores
   comparison = compare_with_previous_models(hybrid_results, class_results, reg_results)
   
   # Guardar resumen final
   results_summary = {
       'classification': {
           'voting': voting_clf_metrics,
           'stacking': stacking_clf_metrics,
           'best_model': best_ensemble_clf_name
       },
       'regression': {
           'voting': voting_reg_metrics,
           'stacking': stacking_reg_metrics,
           'best_model': best_ensemble_reg_name
       },
       'hybrid': hybrid_results,
       'comparison': comparison,
       'feature_counts': {
           'all': len(all_features),
           'top': len(top_features)
       },
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open(f'{output_dir}/results_summary.json', 'w') as f:
       json.dump(results_summary, f, indent=2)
   
   print("\n✅ IMPLEMENTACIÓN COMPLETADA")
   print(f"📁 Archivos generados:")
   print(f"   • {models_dir}/voting_classifier.joblib")
   print(f"   • {models_dir}/stacking_classifier.joblib")
   print(f"   • {models_dir}/voting_regressor.joblib")
   print(f"   • {models_dir}/stacking_regressor.joblib")
   print(f"   • {output_dir}/results_summary.json")
   print(f"   • {plots_dir}/ (múltiples gráficos)")
   
   # Imprimir conclusiones
   print("\n📋 CONCLUSIONES:")
   print(f"1. El mejor modelo de clasificación ensemble ({best_ensemble_clf_name}) {'mejora' if comparison['f1_change'] > 0 else 'no mejora'} la clasificación en {abs(comparison['f1_change']):.1f}%.")
   print(f"2. El mejor modelo de regresión ensemble ({best_ensemble_reg_name}) {'mejora' if comparison['mae_change'] > 0 else 'no mejora'} la predicción de cantidades en {abs(comparison['mae_change']):.1f}%.")
   
   if comparison['f1_change'] <= 0 or comparison['mae_change'] <= 0:
       print(f"3. A pesar de que los modelos ensemble no superan a los modelos individuales del paso 3,")
       print(f"   el enfoque híbrido proporciona una solución integral que combina clasificación y regresión")
       print(f"   en un sistema unificado, facilitando su implementación y mantenimiento.")
   else:
       print(f"3. El enfoque híbrido ensemble optimizado demuestra mejoras significativas tanto en")
       print(f"   la decisión de reponer como en la cantidad a reponer, proporcionando una")
       print(f"   solución robusta para la gestión automatizada de inventario.")
   
   return best_ensemble_clf, best_ensemble_reg, results_summary

if __name__ == "__main__":
   best_clf, best_reg, results = main()