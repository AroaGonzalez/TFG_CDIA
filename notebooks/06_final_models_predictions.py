# notebooks/06_final_models_predictions.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
import xgboost as xgb
import lightgbm as lgb

def load_validation_results():
    """Cargar resultados de validación y datos"""
    try:
        # Datos procesados
        df = pd.read_csv('data/processed/features_engineered.csv')
        
        # Resultados de validación
        validation_summary = pd.read_csv('results/final_validation_summary.csv')
        
        # Features consensus
        consensus_ranking = pd.read_csv('results/feature_consensus_ranking.csv', index_col=0)
        
        print(f"✅ Datos cargados: {df.shape}")
        print(f"✅ Modelos validados: {len(validation_summary)}")
        
        return df, validation_summary, consensus_ranking
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None, None

def select_best_models(validation_summary):
    """Seleccionar mejores modelos basado en validación"""
    print("\n🏆 SELECCIONANDO MEJORES MODELOS")
    print("-" * 30)
    
    # Ranking por clasificación (F1-Score)
    best_classification = validation_summary.loc[
        validation_summary['classification_test_f1'].idxmax()
    ]
    
    # Ranking por regresión (R²)
    best_regression = validation_summary.loc[
        validation_summary['regression_test_r2'].idxmax()
    ]
    
    # Modelo más balanceado
    validation_summary['combined_score'] = (
        validation_summary['classification_test_f1'] + 
        validation_summary['regression_test_r2'].clip(lower=0)
    ) / 2
    
    best_combined = validation_summary.loc[
        validation_summary['combined_score'].idxmax()
    ]
    
    print(f"🎯 Mejor Clasificación: {best_classification['model']} (F1: {best_classification['classification_test_f1']:.3f})")
    print(f"📈 Mejor Regresión: {best_regression['model']} (R²: {best_regression['regression_test_r2']:.3f})")
    print(f"⚖️ Mejor Combinado: {best_combined['model']} (Score: {best_combined['combined_score']:.3f})")
    
    return {
        'classification': best_classification['model'],
        'regression': best_regression['model'],
        'combined': best_combined['model']
    }

def prepare_final_datasets(df, consensus_ranking, n_features=15):
    """Preparar datasets finales con mejores features"""
    print(f"\n📊 PREPARANDO DATASETS FINALES")
    print("-" * 30)
    
    # Top features
    top_features = consensus_ranking.sort_values('mean_rank').head(n_features).index.tolist()
    
    # Preparar datos
    X = df[top_features].fillna(df[top_features].median())
    y_class = df['necesita_reposicion']
    y_reg = df['cantidad_a_reponer']
    
    # Split temporal (80% train, 20% test)
    if 'fecha_recuento' in df.columns:
        df_sorted = df.sort_values('fecha_recuento')
        split_idx = int(len(df_sorted) * 0.8)
        
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train_class, y_test_class = y_class.loc[train_idx], y_class.loc[test_idx]
        y_train_reg, y_test_reg = y_reg.loc[train_idx], y_reg.loc[test_idx]
        
        print("📅 Split temporal aplicado")
    else:
        X_train, X_test, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        y_train_reg = y_reg.loc[X_train.index]
        y_test_reg = y_reg.loc[X_test.index]
        
        print("🎲 Split aleatorio aplicado")
    
    # Para regresión, filtrar casos con reposición > 0
    reg_mask_train = y_train_reg > 0
    reg_mask_test = y_test_reg > 0
    
    X_train_reg = X_train[reg_mask_train]
    X_test_reg = X_test[reg_mask_test]
    y_train_reg_filtered = y_train_reg[reg_mask_train]
    y_test_reg_filtered = y_test_reg[reg_mask_test]
    
    print(f"✅ Datasets preparados:")
    print(f"  • Clasificación - Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"  • Regresión - Train: {len(X_train_reg):,}, Test: {len(X_test_reg):,}")
    print(f"  • Features utilizados: {len(top_features)}")
    
    return {
        'classification': {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train_class, 'y_test': y_test_class
        },
        'regression': {
            'X_train': X_train_reg, 'X_test': X_test_reg,
            'y_train': y_train_reg_filtered, 'y_test': y_test_reg_filtered
        },
        'features': top_features
    }

def create_optimized_models(best_models):
    """Crear modelos con hiperparámetros optimizados"""
    print(f"\n🔧 CREANDO MODELOS OPTIMIZADOS")
    print("-" * 30)
    
    models = {}
    
    # Random Forest optimizado
    models['Random Forest'] = {
        'classification': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'regression': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }
    
    # XGBoost optimizado
    models['XGBoost'] = {
        'classification': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'regression': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # LightGBM optimizado
    models['LightGBM'] = {
        'classification': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=50,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        'regression': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=50,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    }
    
    print(f"✅ Modelos creados: {list(models.keys())}")
    return models

def create_ensemble_models(individual_models, datasets):
    """Crear modelos ensemble"""
    print(f"\n🤝 CREANDO MODELOS ENSEMBLE")
    print("-" * 30)
    
    # Ensemble de clasificación
    classification_estimators = [
        ('rf', individual_models['Random Forest']['classification']),
        ('xgb', individual_models['XGBoost']['classification']),
        ('lgb', individual_models['LightGBM']['classification'])
    ]
    
    ensemble_classification = VotingClassifier(
        estimators=classification_estimators,
        voting='soft',  # Usar probabilidades
        n_jobs=-1
    )
    
    # Ensemble de regresión
    regression_estimators = [
        ('rf', individual_models['Random Forest']['regression']),
        ('xgb', individual_models['XGBoost']['regression']),
        ('lgb', individual_models['LightGBM']['regression'])
    ]
    
    ensemble_regression = VotingRegressor(
        estimators=regression_estimators,
        n_jobs=-1
    )
    
    print("✅ Modelos ensemble creados")
    
    return {
        'classification': ensemble_classification,
        'regression': ensemble_regression
    }

def train_and_evaluate_final_models(models, ensemble_models, datasets):
    """Entrenar y evaluar modelos finales"""
    print(f"\n🚀 ENTRENANDO Y EVALUANDO MODELOS FINALES")
    print("="*50)
    
    final_results = {}
    trained_models = {}
    
    # Datos de entrenamiento y test
    class_data = datasets['classification']
    reg_data = datasets['regression']
    
    # 1. Entrenar modelos individuales
    for model_name, model_dict in models.items():
        print(f"\n🔍 Entrenando {model_name}...")
        
        # Clasificación
        model_class = model_dict['classification']
        model_class.fit(class_data['X_train'], class_data['y_train'])
        y_pred_class = model_class.predict(class_data['X_test'])
        
        # Métricas clasificación
        class_acc = accuracy_score(class_data['y_test'], y_pred_class)
        class_f1 = f1_score(class_data['y_test'], y_pred_class)
        
        # Regresión
        model_reg = model_dict['regression']
        model_reg.fit(reg_data['X_train'], reg_data['y_train'])
        y_pred_reg = model_reg.predict(reg_data['X_test'])
        y_pred_reg = np.maximum(0, y_pred_reg)  # No valores negativos
        
        # Métricas regresión
        reg_r2 = r2_score(reg_data['y_test'], y_pred_reg)
        reg_mae = mean_absolute_error(reg_data['y_test'], y_pred_reg)
        
        print(f"  📊 Clasificación: Accuracy={class_acc:.3f}, F1={class_f1:.3f}")
        print(f"  📊 Regresión: R²={reg_r2:.3f}, MAE={reg_mae:.2f}")
        
        # Guardar resultados
        final_results[model_name] = {
            'classification': {'accuracy': class_acc, 'f1': class_f1},
            'regression': {'r2': reg_r2, 'mae': reg_mae}
        }
        
        trained_models[model_name] = {
            'classification': model_class,
            'regression': model_reg
        }
    
    # 2. Entrenar modelos ensemble
    print(f"\n🤝 Entrenando Ensemble...")
    
    # Ensemble clasificación
    ensemble_class = ensemble_models['classification']
    ensemble_class.fit(class_data['X_train'], class_data['y_train'])
    y_pred_ensemble_class = ensemble_class.predict(class_data['X_test'])
    
    ensemble_class_acc = accuracy_score(class_data['y_test'], y_pred_ensemble_class)
    ensemble_class_f1 = f1_score(class_data['y_test'], y_pred_ensemble_class)
    
    # Ensemble regresión
    ensemble_reg = ensemble_models['regression']
    ensemble_reg.fit(reg_data['X_train'], reg_data['y_train'])
    y_pred_ensemble_reg = ensemble_reg.predict(reg_data['X_test'])
    y_pred_ensemble_reg = np.maximum(0, y_pred_ensemble_reg)
    
    ensemble_reg_r2 = r2_score(reg_data['y_test'], y_pred_ensemble_reg)
    ensemble_reg_mae = mean_absolute_error(reg_data['y_test'], y_pred_ensemble_reg)
    
    print(f"  📊 Ensemble Clasificación: Accuracy={ensemble_class_acc:.3f}, F1={ensemble_class_f1:.3f}")
    print(f"  📊 Ensemble Regresión: R²={ensemble_reg_r2:.3f}, MAE={ensemble_reg_mae:.2f}")
    
    # Guardar ensemble
    final_results['Ensemble'] = {
        'classification': {'accuracy': ensemble_class_acc, 'f1': ensemble_class_f1},
        'regression': {'r2': ensemble_reg_r2, 'mae': ensemble_reg_mae}
    }
    
    trained_models['Ensemble'] = {
        'classification': ensemble_class,
        'regression': ensemble_reg
    }
    
    return final_results, trained_models

def generate_stock_predictions(trained_models, datasets, df_original):
    """Generar predicciones de stock teórico para todo el dataset"""
    print(f"\n🎯 GENERANDO PREDICCIONES DE STOCK TEÓRICO")
    print("-" * 30)
    
    # Usar el mejor modelo (Ensemble o individual)
    best_model_name = 'Ensemble'  # Por defecto ensemble
    
    # Preparar datos completos
    features = datasets['features']
    X_full = df_original[features].fillna(df_original[features].median())
    
    # Modelo de clasificación (¿necesita reposición?)
    model_class = trained_models[best_model_name]['classification']
    necesita_reposicion_pred = model_class.predict(X_full)
    necesita_reposicion_proba = model_class.predict_proba(X_full)[:, 1]
    
    # Modelo de regresión (¿cuánto reponer?)
    model_reg = trained_models[best_model_name]['regression']
    
    # Para regresión, predecir solo casos que necesitan reposición
    cantidad_pred = np.zeros(len(X_full))
    mask_necesita = necesita_reposicion_pred == 1
    
    if mask_necesita.sum() > 0:
        cantidad_pred[mask_necesita] = model_reg.predict(X_full[mask_necesita])
        cantidad_pred = np.maximum(0, cantidad_pred)  # No valores negativos
    
    # Crear DataFrame con predicciones
    predictions_df = df_original[['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'STOCK_RECUENTOS', 'CAPACIDAD_MAXIMA']].copy()
    predictions_df['STOCK_TEORICO_PREDICHO'] = predictions_df['STOCK_RECUENTOS'] + cantidad_pred
    predictions_df['NECESITA_REPOSICION'] = necesita_reposicion_pred
    predictions_df['PROBABILIDAD_REPOSICION'] = necesita_reposicion_proba
    predictions_df['CANTIDAD_A_REPONER'] = cantidad_pred
    
    # Calcular stock teórico como % de capacidad
    predictions_df['OCUPACION_ACTUAL'] = predictions_df['STOCK_RECUENTOS'] / predictions_df['CAPACIDAD_MAXIMA']
    predictions_df['OCUPACION_PREDICHA'] = predictions_df['STOCK_TEORICO_PREDICHO'] / predictions_df['CAPACIDAD_MAXIMA']
    
    # Estadísticas
    casos_reposicion = mask_necesita.sum()
    cantidad_total = cantidad_pred.sum()
    cantidad_promedio = cantidad_pred[mask_necesita].mean() if casos_reposicion > 0 else 0
    
    print(f"✅ Predicciones generadas:")
    print(f"  • Total registros: {len(predictions_df):,}")
    print(f"  • Casos que necesitan reposición: {casos_reposicion:,} ({casos_reposicion/len(predictions_df)*100:.1f}%)")
    print(f"  • Cantidad total a reponer: {cantidad_total:.0f} unidades")
    print(f"  • Cantidad promedio por caso: {cantidad_promedio:.1f} unidades")
    
    return predictions_df

def save_final_models_and_predictions(trained_models, predictions_df, datasets, final_results):
    """Guardar modelos entrenados y predicciones"""
    print(f"\n💾 GUARDANDO MODELOS Y PREDICCIONES")
    print("-" * 30)
    
    # Crear directorio para modelos
    os.makedirs('models', exist_ok=True)
    
    # Guardar modelos entrenados
    for model_name, models in trained_models.items():
        # Guardar modelo de clasificación
        joblib.dump(
            models['classification'], 
            f'models/{model_name.replace(" ", "_")}_classification.pkl'
        )
        
        # Guardar modelo de regresión
        joblib.dump(
            models['regression'], 
            f'models/{model_name.replace(" ", "_")}_regression.pkl'
        )
    
    # Guardar información de features
    features_info = {
        'features': datasets['features'],
        'n_features': len(datasets['features']),
        'model_type': 'stock_prediction',
        'created_date': datetime.now().isoformat()
    }
    
    import json
    with open('models/features_info.json', 'w') as f:
        json.dump(features_info, f, indent=2)
    
    # Guardar predicciones completas
    predictions_df.to_csv('results/stock_predictions_final.csv', index=False)
    
    # Guardar resumen de resultados finales
    final_summary = []
    for model_name, results in final_results.items():
        final_summary.append({
            'model': model_name,
            'classification_accuracy': results['classification']['accuracy'],
            'classification_f1': results['classification']['f1'],
            'regression_r2': results['regression']['r2'],
            'regression_mae': results['regression']['mae']
        })
    
    pd.DataFrame(final_summary).to_csv('results/final_models_summary.csv', index=False)
    
    print("✅ Archivos guardados:")
    print("  • models/*.pkl (modelos entrenados)")
    print("  • models/features_info.json")
    print("  • results/stock_predictions_final.csv")
    print("  • results/final_models_summary.csv")

def create_final_performance_visualization(final_results, predictions_df):
    """Crear visualización final de rendimiento"""
    print(f"\n📊 CREANDO VISUALIZACIÓN FINAL")
    print("-" * 30)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparación F1-Score clasificación
    models = list(final_results.keys())
    f1_scores = [final_results[m]['classification']['f1'] for m in models]
    
    axes[0,0].bar(models, f1_scores, color='skyblue', edgecolor='black')
    axes[0,0].set_title('F1-Score Clasificación\n(¿Necesita Reposición?)')
    axes[0,0].set_ylabel('F1-Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Comparación R² regresión
    r2_scores = [final_results[m]['regression']['r2'] for m in models]
    
    axes[0,1].bar(models, r2_scores, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('R² Regresión\n(¿Cuánto Reponer?)')
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Distribución de predicciones de reposición
    reposicion_counts = predictions_df['NECESITA_REPOSICION'].value_counts()
    axes[0,2].pie(reposicion_counts.values, 
                  labels=['No necesita', 'Necesita reposición'],
                  autopct='%1.1f%%', colors=['lightgreen', 'orange'])
    axes[0,2].set_title('Distribución Predicciones\n(Necesidad de Reposición)')
    
    # 4. Distribución cantidad a reponer
    cantidad_no_zero = predictions_df[predictions_df['CANTIDAD_A_REPONER'] > 0]['CANTIDAD_A_REPONER']
    axes[1,0].hist(cantidad_no_zero, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Distribución Cantidad a Reponer\n(Casos > 0)')
    axes[1,0].set_xlabel('Unidades a reponer')
    axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Ocupación actual vs predicha
    sample_data = predictions_df.sample(n=min(2000, len(predictions_df)))
    axes[1,1].scatter(sample_data['OCUPACION_ACTUAL'], 
                      sample_data['OCUPACION_PREDICHA'], 
                      alpha=0.6, s=1)
    axes[1,1].plot([0, 1], [0, 1], 'r--', alpha=0.7)
    axes[1,1].set_xlabel('Ocupación Actual')
    axes[1,1].set_ylabel('Ocupación Predicha')
    axes[1,1].set_title('Ocupación Actual vs Predicha')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Probabilidad de reposición
    axes[1,2].hist(predictions_df['PROBABILIDAD_REPOSICION'], 
                   bins=30, alpha=0.7, color='brown', edgecolor='black')
    axes[1,2].set_title('Distribución Probabilidad\nde Reposición')
    axes[1,2].set_xlabel('Probabilidad')
    axes[1,2].set_ylabel('Frecuencia')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/final_models_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 INICIANDO MODELOS FINALES Y PREDICCIONES")
    print("="*60)
    
    # Cargar datos y resultados
    df, validation_summary, consensus_ranking = load_validation_results()
    if df is None:
        return
    
    # Seleccionar mejores modelos
    best_models = select_best_models(validation_summary)
    
    # Preparar datasets finales
    datasets = prepare_final_datasets(df, consensus_ranking)
    
    # Crear modelos optimizados
    models = create_optimized_models(best_models)
    
    # Crear modelos ensemble
    ensemble_models = create_ensemble_models(models, datasets)
    
    # Entrenar y evaluar
    final_results, trained_models = train_and_evaluate_final_models(
        models, ensemble_models, datasets
    )
    
    # Generar predicciones de stock teórico
    predictions_df = generate_stock_predictions(trained_models, datasets, df)
    
    # Guardar todo
    save_final_models_and_predictions(trained_models, predictions_df, datasets, final_results)
    
    # Crear visualizaciones
    create_final_performance_visualization(final_results, predictions_df)
    
    print(f"\n🎉 MODELOS FINALES COMPLETADOS")
    print(f"✅ Modelos entrenados y guardados")
    print(f"✅ Predicciones de stock teórico generadas")
    print(f"✅ Sistema listo para producción")
    
    print(f"\n🎯 PRÓXIMO PASO: 07_business_analysis_simulation.py")
    
    return final_results, trained_models, predictions_df

if __name__ == "__main__":
    final_results, trained_models, predictions_df = main()