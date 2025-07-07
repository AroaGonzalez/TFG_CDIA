# notebooks/05_final_validation_evaluation.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit, 
    validation_curve, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def load_optimized_data():
    """Cargar datos y resultados de optimización"""
    try:
        # Datos procesados
        df = pd.read_csv('data/processed/features_engineered.csv')
        
        # Resultados de optimización
        optimization_results = pd.read_csv('results/optimization_results.csv')
        
        # Consensus ranking de features
        consensus_ranking = pd.read_csv('results/feature_consensus_ranking.csv', index_col=0)
        
        print(f"✅ Datos cargados: {df.shape}")
        print(f"✅ Modelos optimizados: {len(optimization_results)}")
        
        return df, optimization_results, consensus_ranking
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Ejecuta primero 04_feature_analysis_optimization.py")
        return None, None, None

def prepare_temporal_validation_data(df):
    """Preparar datos para validación temporal"""
    print("\n📅 PREPARANDO VALIDACIÓN TEMPORAL")
    print("-" * 30)
    
    # Convertir fechas si existen
    if 'fecha_recuento' in df.columns:
        df['fecha_recuento'] = pd.to_datetime(df['fecha_recuento'], errors='coerce')
    else:
        # Generar fechas sintéticas basadas en dias_desde_recuento
        base_date = datetime.now()
        df['fecha_recuento'] = base_date - pd.to_timedelta(df['dias_desde_recuento'], unit='D')
    
    # Ordenar por fecha
    df_sorted = df.sort_values('fecha_recuento').reset_index(drop=True)
    
    # Crear splits temporales
    # 60% train, 20% validation, 20% test
    n_total = len(df_sorted)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.8)
    
    df_train = df_sorted[:n_train].copy()
    df_val = df_sorted[n_train:n_val].copy()
    df_test = df_sorted[n_val:].copy()
    
    print(f"📊 Split temporal:")
    print(f"  • Train: {len(df_train):,} registros ({df_train['fecha_recuento'].min().date()} a {df_train['fecha_recuento'].max().date()})")
    print(f"  • Validation: {len(df_val):,} registros ({df_val['fecha_recuento'].min().date()} a {df_val['fecha_recuento'].max().date()})")
    print(f"  • Test: {len(df_test):,} registros ({df_test['fecha_recuento'].min().date()} a {df_test['fecha_recuento'].max().date()})")
    
    return df_train, df_val, df_test

def get_top_features(consensus_ranking, n_features=15):
    """Obtener top features del consensus ranking"""
    top_features = consensus_ranking.sort_values('mean_rank').head(n_features).index.tolist()
    print(f"🎯 Usando top {n_features} features:")
    for i, feature in enumerate(top_features, 1):
        rank = consensus_ranking.loc[feature, 'mean_rank']
        print(f"  {i:2d}. {feature:25s} (rank: {rank:.1f})")
    
    return top_features

def temporal_cross_validation(X, y, model, task_type='classification', n_splits=5):
    """Validación cruzada temporal"""
    print(f"\n⏰ VALIDACIÓN CRUZADA TEMPORAL - {task_type.upper()}")
    print("-" * 30)
    
    # TimeSeriesSplit para validación temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar modelo
        model.fit(X_train_fold, y_train_fold)
        
        # Predicciones
        y_pred = model.predict(X_val_fold)
        
        # Métricas según tipo de tarea
        if task_type == 'classification':
            score = f1_score(y_val_fold, y_pred)
            metric_name = 'F1-Score'
        else:  # regression
            y_pred = np.maximum(0, y_pred)  # No permitir predicciones negativas
            score = r2_score(y_val_fold, y_pred)
            metric_name = 'R²'
        
        scores.append(score)
        fold_details.append({
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'score': score
        })
        
        print(f"  Fold {fold}: {metric_name} = {score:.3f} (train: {len(train_idx):,}, val: {len(val_idx):,})")
    
    print(f"\n📊 Resultado CV Temporal:")
    print(f"  • {metric_name} promedio: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
    print(f"  • Rango: {np.min(scores):.3f} - {np.max(scores):.3f}")
    
    return scores, fold_details

def detailed_error_analysis(y_true, y_pred, task_type='regression', segment_data=None):
    """Análisis detallado de errores"""
    print(f"\n🔍 ANÁLISIS DETALLADO DE ERRORES - {task_type.upper()}")
    print("-" * 30)
    
    if task_type == 'regression':
        # Asegurar predicciones no negativas
        y_pred = np.maximum(0, y_pred)
        
        # Calcular errores
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors) / np.maximum(y_true, 1) * 100
        
        # Estadísticas básicas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(pct_errors)
        
        print(f"📈 MÉTRICAS GENERALES:")
        print(f"  • MAE: {mae:.2f}")
        print(f"  • RMSE: {rmse:.2f}")
        print(f"  • R²: {r2:.3f}")
        print(f"  • MAPE: {mape:.1f}%")
        
        # Análisis de distribución de errores
        print(f"\n📊 DISTRIBUCIÓN DE ERRORES:")
        print(f"  • Error medio: {np.mean(errors):.2f}")
        print(f"  • Error mediano: {np.median(errors):.2f}")
        print(f"  • Desviación estándar: {np.std(errors):.2f}")
        print(f"  • Percentil 90 error absoluto: {np.percentile(abs_errors, 90):.2f}")
        print(f"  • Percentil 95 error absoluto: {np.percentile(abs_errors, 95):.2f}")
        
        # Casos problemáticos
        high_error_threshold = np.percentile(abs_errors, 95)
        high_error_cases = abs_errors > high_error_threshold
        
        print(f"\n🚨 CASOS CON ERRORES ALTOS (>P95):")
        print(f"  • Número de casos: {high_error_cases.sum():,} ({high_error_cases.sum()/len(y_true)*100:.1f}%)")
        print(f"  • Error promedio en estos casos: {abs_errors[high_error_cases].mean():.2f}")
        
        return {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'errors': errors, 'abs_errors': abs_errors, 'pct_errors': pct_errors,
            'high_error_cases': high_error_cases
        }
        
    else:  # classification
        # Métricas de clasificación
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"🎯 MÉTRICAS GENERALES:")
        print(f"  • Accuracy: {accuracy:.3f}")
        print(f"  • Precision: {precision:.3f}")
        print(f"  • Recall: {recall:.3f}")
        print(f"  • F1-Score: {f1:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📊 MATRIZ DE CONFUSIÓN:")
        print(f"  • Verdaderos Negativos: {cm[0,0]:,}")
        print(f"  • Falsos Positivos: {cm[0,1]:,}")
        print(f"  • Falsos Negativos: {cm[1,0]:,}")
        print(f"  • Verdaderos Positivos: {cm[1,1]:,}")
        
        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm
        }

def segment_analysis(df_test, y_true, y_pred, task_type='regression'):
    """Análisis por segmentos (alias, tiendas, etc.)"""
    print(f"\n🏪 ANÁLISIS POR SEGMENTOS")
    print("-" * 30)
    
    results = {}
    
    # Análisis por Alias
    if 'ID_ALIAS' in df_test.columns:
        print("📊 Análisis por ALIAS:")
        alias_results = []
        
        for alias_id in df_test['ID_ALIAS'].unique()[:10]:  # Top 10 alias con más datos
            mask = df_test['ID_ALIAS'] == alias_id
            if mask.sum() < 20:  # Skip alias con pocos datos
                continue
                
            y_true_alias = y_true[mask]
            y_pred_alias = y_pred[mask]
            
            if task_type == 'regression':
                y_pred_alias = np.maximum(0, y_pred_alias)
                mae = mean_absolute_error(y_true_alias, y_pred_alias)
                r2 = r2_score(y_true_alias, y_pred_alias)
                alias_results.append({
                    'ID_ALIAS': alias_id,
                    'n_records': mask.sum(),
                    'MAE': mae,
                    'R2': r2
                })
            else:
                f1 = f1_score(y_true_alias, y_pred_alias)
                accuracy = accuracy_score(y_true_alias, y_pred_alias)
                alias_results.append({
                    'ID_ALIAS': alias_id,
                    'n_records': mask.sum(),
                    'F1': f1,
                    'Accuracy': accuracy
                })
        
        if alias_results:
            alias_df = pd.DataFrame(alias_results)
            if task_type == 'regression':
                alias_df = alias_df.sort_values('R2', ascending=False)
                print(f"  Top 5 Alias por R²:")
                for _, row in alias_df.head().iterrows():
                    print(f"    Alias {row['ID_ALIAS']:3.0f}: R²={row['R2']:.3f}, MAE={row['MAE']:.1f} ({row['n_records']:,} registros)")
            else:
                alias_df = alias_df.sort_values('F1', ascending=False)
                print(f"  Top 5 Alias por F1:")
                for _, row in alias_df.head().iterrows():
                    print(f"    Alias {row['ID_ALIAS']:3.0f}: F1={row['F1']:.3f}, Acc={row['Accuracy']:.3f} ({row['n_records']:,} registros)")
            
            results['alias_analysis'] = alias_df
    
    # Análisis por tamaño de tienda
    if 'tienda_tamaño' in df_test.columns:
        print(f"\n📊 Análisis por TAMAÑO DE TIENDA:")
        for tamaño in df_test['tienda_tamaño'].unique():
            mask = df_test['tienda_tamaño'] == tamaño
            if mask.sum() < 50:  # Skip categorías con pocos datos
                continue
                
            y_true_tam = y_true[mask]
            y_pred_tam = y_pred[mask]
            
            if task_type == 'regression':
                y_pred_tam = np.maximum(0, y_pred_tam)
                mae = mean_absolute_error(y_true_tam, y_pred_tam)
                r2 = r2_score(y_true_tam, y_pred_tam)
                print(f"  {tamaño:10s}: R²={r2:.3f}, MAE={mae:.1f} ({mask.sum():,} registros)")
            else:
                f1 = f1_score(y_true_tam, y_pred_tam)
                accuracy = accuracy_score(y_true_tam, y_pred_tam)
                print(f"  {tamaño:10s}: F1={f1:.3f}, Acc={accuracy:.3f} ({mask.sum():,} registros)")
    
    # Análisis por nivel de urgencia
    if 'urgencia_reposicion' in df_test.columns:
        print(f"\n📊 Análisis por URGENCIA DE REPOSICIÓN:")
        urgencia_map = {0: 'Normal', 1: 'Medio', 2: 'Alto', 3: 'Crítico'}
        
        for urgencia in sorted(df_test['urgencia_reposicion'].unique()):
            mask = df_test['urgencia_reposicion'] == urgencia
            if mask.sum() < 20:
                continue
                
            y_true_urg = y_true[mask]
            y_pred_urg = y_pred[mask]
            urgencia_name = urgencia_map.get(urgencia, f'Nivel_{urgencia}')
            
            if task_type == 'regression':
                y_pred_urg = np.maximum(0, y_pred_urg)
                mae = mean_absolute_error(y_true_urg, y_pred_urg)
                r2 = r2_score(y_true_urg, y_pred_urg)
                print(f"  {urgencia_name:8s}: R²={r2:.3f}, MAE={mae:.1f} ({mask.sum():,} registros)")
            else:
                f1 = f1_score(y_true_urg, y_pred_urg)
                accuracy = accuracy_score(y_true_urg, y_pred_urg)
                print(f"  {urgencia_name:8s}: F1={f1:.3f}, Acc={accuracy:.3f} ({mask.sum():,} registros)")
    
    return results

def create_validation_visualizations(df_test, y_true_class, y_pred_class, y_true_reg, y_pred_reg, error_analysis_reg):
    """Crear visualizaciones de validación"""
    print(f"\n📊 CREANDO VISUALIZACIONES DE VALIDACIÓN")
    print("-" * 30)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    # 1. Matriz de confusión - Clasificación
    cm = confusion_matrix(y_true_class, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Matriz de Confusión\n(Necesita Reposición)')
    axes[0,0].set_xlabel('Predicción')
    axes[0,0].set_ylabel('Realidad')
    
    # 2. Distribución de probabilidades - Clasificación
    if hasattr(y_pred_class, 'predict_proba'):  # Si tenemos probabilidades
        # Esto requeriría el modelo, por ahora skip
        pass
    else:
        # Mostrar distribución de predicciones por clase real
        df_class_viz = pd.DataFrame({
            'real': y_true_class,
            'pred': y_pred_class
        })
        df_class_viz['correct'] = df_class_viz['real'] == df_class_viz['pred']
        df_class_viz['correct'].value_counts().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Predicciones Correctas vs Incorrectas')
        axes[0,1].set_xlabel('¿Predicción Correcta?')
        axes[0,1].set_ylabel('Número de casos')
    
    # 3. Predicciones vs Realidad - Regresión
    y_pred_reg_clean = np.maximum(0, y_pred_reg)
    axes[0,2].scatter(y_true_reg, y_pred_reg_clean, alpha=0.6, s=1)
    max_val = max(y_true_reg.max(), y_pred_reg_clean.max())
    axes[0,2].plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    axes[0,2].set_xlabel('Stock Real')
    axes[0,2].set_ylabel('Stock Predicho')
    axes[0,2].set_title('Predicciones vs Realidad\n(Regresión)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Distribución de errores - Regresión
    errors = error_analysis_reg['errors']
    axes[0,3].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0,3].axvline(0, color='red', linestyle='--', label='Error = 0')
    axes[0,3].set_xlabel('Error (Real - Predicho)')
    axes[0,3].set_ylabel('Frecuencia')
    axes[0,3].set_title('Distribución de Errores')
    axes[0,3].legend()
    axes[0,3].grid(True, alpha=0.3)
    
    # 5. Errores absolutos por valor real
    abs_errors = error_analysis_reg['abs_errors']
    axes[1,0].scatter(y_true_reg, abs_errors, alpha=0.6, s=1)
    axes[1,0].set_xlabel('Stock Real')
    axes[1,0].set_ylabel('Error Absoluto')
    axes[1,0].set_title('Error Absoluto vs Stock Real')
    axes[1,0].grid(True, alpha=0.3)
    
    # 6. Errores porcentuales
    pct_errors = error_analysis_reg['pct_errors']
    # Filtrar outliers extremos para visualización
    pct_errors_clean = pct_errors[pct_errors <= np.percentile(pct_errors, 95)]
    axes[1,1].hist(pct_errors_clean, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Error Porcentual (%)')
    axes[1,1].set_ylabel('Frecuencia')
    axes[1,1].set_title('Distribución Error Porcentual')
    axes[1,1].grid(True, alpha=0.3)
    
    # 7. Casos con errores altos
    high_error_cases = error_analysis_reg['high_error_cases']
    error_categories = ['Error Normal', 'Error Alto']
    error_counts = [(~high_error_cases).sum(), high_error_cases.sum()]
    axes[1,2].pie(error_counts, labels=error_categories, autopct='%1.1f%%')
    axes[1,2].set_title('Distribución de Casos\npor Nivel de Error')
    
    # 8. Residuos vs predicciones
    axes[1,3].scatter(y_pred_reg_clean, errors, alpha=0.6, s=1)
    axes[1,3].axhline(0, color='red', linestyle='--')
    axes[1,3].set_xlabel('Stock Predicho')
    axes[1,3].set_ylabel('Residuos')
    axes[1,3].set_title('Residuos vs Predicciones')
    axes[1,3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/final_validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_best_models(df_train, df_val, df_test, top_features, optimization_results):
    """Evaluar los mejores modelos con validación temporal"""
    print(f"\n🏆 EVALUACIÓN FINAL DE MEJORES MODELOS")
    print("="*50)
    
    # Preparar datos
    X_train = df_train[top_features].fillna(df_train[top_features].median())
    X_val = df_val[top_features].fillna(df_val[top_features].median())
    X_test = df_test[top_features].fillna(df_test[top_features].median())
    
    y_train_class = df_train['necesita_reposicion']
    y_val_class = df_val['necesita_reposicion']
    y_test_class = df_test['necesita_reposicion']
    
    y_train_reg = df_train['cantidad_a_reponer']
    y_val_reg = df_val['cantidad_a_reponer']
    y_test_reg = df_test['cantidad_a_reponer']
    
    # Para regresión, filtrar solo casos con reposición > 0
    reg_mask_train = y_train_reg > 0
    reg_mask_val = y_val_reg > 0
    reg_mask_test = y_test_reg > 0
    
    X_train_reg = X_train[reg_mask_train]
    X_val_reg = X_val[reg_mask_val]
    X_test_reg = X_test[reg_mask_test]
    
    y_train_reg_filtered = y_train_reg[reg_mask_train]
    y_val_reg_filtered = y_val_reg[reg_mask_val]
    y_test_reg_filtered = y_test_reg[reg_mask_test]
    
    print(f"📊 Datos preparados:")
    print(f"  • Clasificación - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"  • Regresión - Train: {len(X_train_reg):,}, Val: {len(X_val_reg):,}, Test: {len(X_test_reg):,}")
    
    final_results = {}
    final_predictions = {}
    
    # Obtener mejores modelos de optimization_results
    best_models = optimization_results['model'].unique()
    
    for model_name in best_models:
        print(f"\n🔍 Evaluando {model_name}...")
        
        # CLASIFICACIÓN
        if model_name == 'Random Forest':
            model_class = RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=2,
                random_state=42, n_jobs=-1
            )
        elif model_name == 'XGBoost':
            model_class = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='logloss', n_jobs=-1
            )
        elif model_name == 'LightGBM':
            model_class = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1, n_jobs=-1
            )
        else:
            continue
        
        # Entrenar y evaluar clasificación
        model_class.fit(X_train, y_train_class)
        y_pred_class = model_class.predict(X_test)
        
        # Validación cruzada temporal
        cv_scores_class, _ = temporal_cross_validation(
            pd.concat([X_train, X_val]), 
            pd.concat([y_train_class, y_val_class]), 
            model_class, 
            'classification'
        )
        
        # Análisis de errores clasificación
        error_analysis_class = detailed_error_analysis(
            y_test_class, y_pred_class, 'classification'
        )
        
        # REGRESIÓN
        if model_name == 'Random Forest':
            model_reg = RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=2,
                random_state=42, n_jobs=-1
            )
        elif model_name == 'XGBoost':
            model_reg = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
        elif model_name == 'LightGBM':
            model_reg = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1, n_jobs=-1
            )
        
        # Entrenar y evaluar regresión
        model_reg.fit(X_train_reg, y_train_reg_filtered)
        y_pred_reg = model_reg.predict(X_test_reg)
        
        # Validación cruzada temporal
        cv_scores_reg, _ = temporal_cross_validation(
            pd.concat([X_train_reg, X_val_reg]), 
            pd.concat([y_train_reg_filtered, y_val_reg_filtered]), 
            model_reg, 
            'regression'
        )
        
        # Análisis de errores regresión
        error_analysis_reg = detailed_error_analysis(
            y_test_reg_filtered, y_pred_reg, 'regression'
        )
        
        # Análisis por segmentos
        segment_results_class = segment_analysis(
            df_test, y_test_class, y_pred_class, 'classification'
        )
        
        segment_results_reg = segment_analysis(
            df_test.loc[reg_mask_test], y_test_reg_filtered, y_pred_reg, 'regression'
        )
        
        # Guardar resultados
        final_results[model_name] = {
            'classification': {
                'cv_scores': cv_scores_class,
                'error_analysis': error_analysis_class,
                'segment_analysis': segment_results_class
            },
            'regression': {
                'cv_scores': cv_scores_reg,
                'error_analysis': error_analysis_reg,
                'segment_analysis': segment_results_reg
            }
        }
        
        final_predictions[model_name] = {
            'classification': {
                'y_true': y_test_class,
                'y_pred': y_pred_class,
                'model': model_class
            },
            'regression': {
                'y_true': y_test_reg_filtered,
                'y_pred': y_pred_reg,
                'model': model_reg
            }
        }
    
    return final_results, final_predictions

def save_validation_results(final_results, final_predictions):
    """Guardar resultados de validación final"""
    print(f"\n💾 GUARDANDO RESULTADOS DE VALIDACIÓN FINAL")
    print("-" * 30)
    
    # Crear resumen de resultados
    summary_results = []
    
    for model_name, results in final_results.items():
        # Clasificación
        class_cv = np.mean(results['classification']['cv_scores'])
        class_f1 = results['classification']['error_analysis']['f1']
        class_accuracy = results['classification']['error_analysis']['accuracy']
        
        # Regresión
        reg_cv = np.mean(results['regression']['cv_scores'])
        reg_r2 = results['regression']['error_analysis']['r2']
        reg_mae = results['regression']['error_analysis']['mae']
        
        summary_results.append({
            'model': model_name,
            'classification_cv_f1': class_cv,
            'classification_test_f1': class_f1,
            'classification_test_accuracy': class_accuracy,
            'regression_cv_r2': reg_cv,
            'regression_test_r2': reg_r2,
            'regression_test_mae': reg_mae
        })
    
    # Guardar resumen
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/final_validation_summary.csv', index=False)
    
    # Guardar predicciones para análisis posterior
    for model_name, predictions in final_predictions.items():
        # Guardar predicciones de clasificación
       class_pred_df = pd.DataFrame({
           'y_true': predictions['classification']['y_true'],
           'y_pred': predictions['classification']['y_pred']
       })
       class_pred_df.to_csv(f'results/predictions_classification_{model_name.replace(" ", "_")}.csv', index=False)
       
       # Guardar predicciones de regresión
       reg_pred_df = pd.DataFrame({
           'y_true': predictions['regression']['y_true'],
           'y_pred': predictions['regression']['y_pred']
       })
       reg_pred_df.to_csv(f'results/predictions_regression_{model_name.replace(" ", "_")}.csv', index=False)
   
    print("✅ Archivos guardados:")
    print(" • results/final_validation_summary.csv")
    print(" • results/predictions_classification_*.csv")
    print(" • results/predictions_regression_*.csv")

def print_final_validation_summary(final_results):
   """Imprimir resumen final de validación"""
   print(f"\n🏆 RESUMEN FINAL DE VALIDACIÓN")
   print("="*60)
   
   for model_name, results in final_results.items():
       print(f"\n🤖 {model_name.upper()}")
       print("-" * 40)
       
       # Clasificación
       class_results = results['classification']
       cv_f1 = np.mean(class_results['cv_scores'])
       cv_f1_std = np.std(class_results['cv_scores'])
       test_f1 = class_results['error_analysis']['f1']
       test_acc = class_results['error_analysis']['accuracy']
       
       print(f"🎯 CLASIFICACIÓN (¿Necesita reposición?):")
       print(f"  • CV F1-Score: {cv_f1:.3f} (±{cv_f1_std:.3f})")
       print(f"  • Test F1-Score: {test_f1:.3f}")
       print(f"  • Test Accuracy: {test_acc:.3f}")
       
       # Regresión
       reg_results = results['regression']
       cv_r2 = np.mean(reg_results['cv_scores'])
       cv_r2_std = np.std(reg_results['cv_scores'])
       test_r2 = reg_results['error_analysis']['r2']
       test_mae = reg_results['error_analysis']['mae']
       test_mape = reg_results['error_analysis']['mape']
       
       print(f"\n📈 REGRESIÓN (¿Cuánto reponer?):")
       print(f"  • CV R²: {cv_r2:.3f} (±{cv_r2_std:.3f})")
       print(f"  • Test R²: {test_r2:.3f}")
       print(f"  • Test MAE: {test_mae:.2f} unidades")
       print(f"  • Test MAPE: {test_mape:.1f}%")
   
   # Determinar mejor modelo general
   print(f"\n🥇 MEJOR MODELO GENERAL:")
   best_model = None
   best_score = -np.inf
   
   for model_name, results in final_results.items():
       # Puntuación combinada (normalizada)
       f1_score = results['classification']['error_analysis']['f1']
       r2_score = results['regression']['error_analysis']['r2']
       combined_score = (f1_score + max(0, r2_score)) / 2  # Promedio simple
       
       print(f"  • {model_name}: {combined_score:.3f} (F1: {f1_score:.3f}, R²: {r2_score:.3f})")
       
       if combined_score > best_score:
           best_score = combined_score
           best_model = model_name
   
   print(f"\n🏆 GANADOR: {best_model} (Score: {best_score:.3f})")

def main():
   print("🚀 INICIANDO VALIDACIÓN FINAL Y EVALUACIÓN PROFUNDA")
   print("="*60)
   
   # Cargar datos y resultados
   df, optimization_results, consensus_ranking = load_optimized_data()
   if df is None:
       return
   
   # Preparar split temporal
   df_train, df_val, df_test = prepare_temporal_validation_data(df)
   
   # Obtener top features
   top_features = get_top_features(consensus_ranking, n_features=15)
   
   # Evaluación final de mejores modelos
   final_results, final_predictions = evaluate_best_models(
       df_train, df_val, df_test, top_features, optimization_results
   )
   
   # Crear visualizaciones de validación
   if final_predictions:
       # Usar el primer modelo para visualizaciones
       first_model = list(final_predictions.keys())[0]
       y_true_class = final_predictions[first_model]['classification']['y_true']
       y_pred_class = final_predictions[first_model]['classification']['y_pred']
       y_true_reg = final_predictions[first_model]['regression']['y_true']
       y_pred_reg = final_predictions[first_model]['regression']['y_pred']
       
       error_analysis_reg = final_results[first_model]['regression']['error_analysis']
       
       create_validation_visualizations(
           df_test, y_true_class, y_pred_class, y_true_reg, y_pred_reg, error_analysis_reg
       )
   
   # Guardar resultados
   save_validation_results(final_results, final_predictions)
   
   # Imprimir resumen final
   print_final_validation_summary(final_results)
   
   print(f"\n✅ VALIDACIÓN FINAL COMPLETADA")
   print(f"📁 Archivos generados:")
   print(f"  • results/final_validation_summary.csv")
   print(f"  • results/predictions_*.csv")
   print(f"  • results/plots/final_validation_analysis.png")
   
   print(f"\n🎯 PRÓXIMO PASO: 06_final_models_predictions.py")
   print(f"  • Entrenar modelos finales con mejores configuraciones")
   print(f"  • Generar predicciones de stock teórico")
   print(f"  • Crear sistema de ensemble")
   
   return final_results, final_predictions, top_features

if __name__ == "__main__":
   final_results, final_predictions, top_features = main()