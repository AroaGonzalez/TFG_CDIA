# 07_final_models_predictions.py - VERSIÓN CON RE-ENTRENAMIENTO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import os
import json
from datetime import datetime
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Configuración
output_dir = 'results/07_final_models'
plots_dir = f'{output_dir}/plots'
final_dir = 'models/final'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

class OptimizedStockPredictor:
    """Predictor híbrido optimizado - entrena modelos frescos con configuración ganadora"""
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        self.is_trained = False
        
    def fit(self, X, y_class, y_reg_log, feature_names=None):
        """Entrenar modelos con configuración ganadora del Script 3"""
        print("🔧 ENTRENANDO MODELOS CON CONFIGURACIÓN GANADORA...")
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split estratificado (igual que Script 3)
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        y_reg_log_train = y_reg_log.iloc[X_train.index]
        y_reg_log_test = y_reg_log.iloc[X_test.index]
        
        # Escalado robusto
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. ENTRENAR CLASIFICADOR - Gradient Boosting (ganador Script 3)
        print("   🎯 Entrenando Gradient Boosting Classifier...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            learning_rate=0.1
        )
        
        # Entrenar con datos sin escalar (como Script 3)
        self.classifier.fit(X_train, y_class_train)
        
        # Evaluar clasificador
        y_class_pred = self.classifier.predict(X_test)
        y_class_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        class_acc = accuracy_score(y_class_test, y_class_pred)
        class_f1 = f1_score(y_class_test, y_class_pred)
        class_auc = roc_auc_score(y_class_test, y_class_proba)
        
        print(f"      ✅ Accuracy: {class_acc:.4f}")
        print(f"      ✅ F1-Score: {class_f1:.4f}")
        print(f"      ✅ ROC-AUC: {class_auc:.4f}")
        
        # 2. ENTRENAR REGRESOR - LightGBM (ganador Script 3)
        print("   📈 Entrenando LightGBM Regressor...")
        
        # Filtrar solo casos positivos para regresión (como Script 3)
        mask_positive = y_class == 1
        X_reg = X[mask_positive]
        y_reg_log_pos = y_reg_log[mask_positive]
        
        print(f"      📊 Casos para regresión: {len(X_reg)}")
        
        # Split para regresión
        X_reg_train, X_reg_test, y_reg_log_train_pos, y_reg_log_test_pos = train_test_split(
            X_reg, y_reg_log_pos, test_size=0.2, random_state=42
        )
        
        # Escalar datos de regresión
        scaler_reg = RobustScaler()
        X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
        X_reg_test_scaled = scaler_reg.transform(X_reg_test)
        
        self.regressor = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # Entrenar con datos escalados
        self.regressor.fit(X_reg_train_scaled, y_reg_log_train_pos)
        
        # Evaluar regresor
        y_reg_pred_log = self.regressor.predict(X_reg_test_scaled)
        y_reg_pred = np.expm1(y_reg_pred_log)
        y_reg_true = np.expm1(y_reg_log_test_pos)
        
        reg_mae = mean_absolute_error(y_reg_true, y_reg_pred)
        reg_r2 = r2_score(y_reg_true, y_reg_pred)
        
        print(f"      ✅ MAE: {reg_mae:.2f}")
        print(f"      ✅ R²: {reg_r2:.4f}")
        
        # 3. OPTIMIZAR UMBRAL
        print("   🎯 Optimizando umbral de clasificación...")
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred_thresh = (y_class_proba > threshold).astype(int)
            f1_thresh = f1_score(y_class_test, y_pred_thresh)
            
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"      ✅ Mejor umbral: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        # Guardar scaler de regresión como atributo
        self.reg_scaler = scaler_reg
        
        self.is_trained = True
        
        return {
            'classification': {
                'accuracy': class_acc,
                'f1_score': class_f1,
                'roc_auc': class_auc,
                'optimized_f1': best_f1,
                'best_threshold': best_threshold
            },
            'regression': {
                'mae': reg_mae,
                'r2': reg_r2,
                'cases_trained': len(X_reg_train)
            }
        }
        
    def predict(self, X):
        """Predicciones híbridas"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Convertir a array
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Limpiar datos
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predicciones de clasificación (sin escalar, como fue entrenado)
        y_class_proba = self.classifier.predict_proba(X_clean)[:, 1]
        y_class_pred = (y_class_proba > self.threshold).astype(int)
        
        # Predicciones de regresión
        y_reg_pred = np.zeros(len(X_clean))
        
        positive_mask = y_class_pred == 1
        if positive_mask.sum() > 0:
            # Escalar solo para regresión
            X_reg_scaled = self.reg_scaler.transform(X_clean[positive_mask])
            
            # Predicir en escala log
            y_reg_log = self.regressor.predict(X_reg_scaled)
            
            # Convertir a escala original
            y_reg_original = np.expm1(y_reg_log)
            y_reg_original = np.clip(y_reg_original, 0, 50000)
            
            y_reg_pred[positive_mask] = y_reg_original
        
        return {
            'necesita_reposicion': y_class_pred,
            'probabilidad_reposicion': y_class_proba,
            'cantidad_a_reponer': y_reg_pred
        }

def load_data():
    """Cargar y preparar datos exactamente como Script 3"""
    print("📦 CARGANDO Y PREPARANDO DATOS")
    print("-" * 40)
    
    df = pd.read_csv('data/processed/02_features/features_engineered.csv')
    print(f"✅ Datos cargados: {df.shape}")
    
    # Cargar features
    with open('results/02_feature_engineering/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
        features = metadata.get('features', [])
    
    # Preparar X (replicar Script 3)
    X_clean = df[features].copy()
    numeric_cols = X_clean.select_dtypes(include=['number']).columns.tolist()
    X = X_clean[numeric_cols].copy()
    
    # Imputar NaN exactamente como Script 3
    X = X.fillna(X.median())
    X = X.fillna(0)
    
    print(f"✅ Features: {X.shape[1]} (debe ser 33 como Script 3)")
    
    if X.shape[1] != 33:
        print(f"⚠️ ADVERTENCIA: Features diferentes a Script 3")
    
    # Targets
    y_class = df['necesita_reposicion'].copy()
    y_reg = df['cantidad_a_reponer'].copy()
    y_reg_log = np.log1p(y_reg)
    
    print(f"✅ Balance: {y_class.mean():.1%} necesitan reposición")
    
    return X, y_class, y_reg, y_reg_log, X.columns.tolist()

def evaluate_predictor(predictor, X, y_class, y_reg):
    """Evaluar predictor en datos de test"""
    print("\n🔍 EVALUACIÓN FINAL")
    print("-" * 40)
    
    # Split para evaluación
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    y_reg_train = y_reg.iloc[X_train.index]
    y_reg_test = y_reg.iloc[X_test.index]
    
    # Predicciones
    predictions = predictor.predict(X_test)
    
    # Métricas de clasificación
    y_class_pred = predictions['necesita_reposicion']
    
    class_acc = accuracy_score(y_class_test, y_class_pred)
    class_prec = precision_score(y_class_test, y_class_pred)
    class_rec = recall_score(y_class_test, y_class_pred)
    class_f1 = f1_score(y_class_test, y_class_pred)
    
    print(f"📊 CLASIFICACIÓN:")
    print(f"   • Accuracy: {class_acc:.4f}")
    print(f"   • Precision: {class_prec:.4f}")
    print(f"   • Recall: {class_rec:.4f}")
    print(f"   • F1-Score: {class_f1:.4f}")
    
    # Métricas de regresión en casos positivos reales
    real_positive = y_class_test == 1
    pred_positive = y_class_pred == 1
    both_positive = real_positive & pred_positive
    
    print(f"\n📈 REGRESIÓN:")
    print(f"   • Casos reales positivos: {real_positive.sum()}")
    print(f"   • Casos predichos positivos: {pred_positive.sum()}")
    print(f"   • Correctamente clasificados: {both_positive.sum()}")
    
    if both_positive.sum() > 5:
        y_reg_pred_pos = predictions['cantidad_a_reponer'][both_positive]
        y_reg_true_pos = y_reg_test[both_positive]
        
        reg_mae = mean_absolute_error(y_reg_true_pos, y_reg_pred_pos)
        reg_r2 = r2_score(y_reg_true_pos, y_reg_pred_pos)
        
        print(f"   • MAE: {reg_mae:.2f}")
        print(f"   • R²: {reg_r2:.4f}")
    else:
        reg_mae = reg_r2 = None
        print(f"   ⚠️ Pocos casos para evaluar regresión")
    
    return {
        'classification': {
            'accuracy': class_acc,
            'precision': class_prec,
            'recall': class_rec,
            'f1_score': class_f1
        },
        'regression': {
            'mae': reg_mae,
            'r2': reg_r2,
            'cases_evaluated': both_positive.sum() if 'both_positive' in locals() else 0
        }
    }, predictions, (X_test, y_class_test, y_reg_test)

def create_visualizations(predictor, test_data, predictions):
    """Crear visualizaciones"""
    print("\n📊 CREANDO VISUALIZACIONES")
    print("-" * 40)
    
    X_test, y_class_test, y_reg_test = test_data
    
    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_class_test, predictions['necesita_reposicion'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Reponer', 'Reponer'],
                yticklabels=['No Reponer', 'Reponer'])
    plt.title('Matriz de Confusión - Predictor Optimizado')
    plt.ylabel('Valores Reales')
    plt.xlabel('Predicciones')
    plt.savefig(f'{plots_dir}/confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribución de probabilidades
    plt.figure(figsize=(10, 6))
    proba_pos = predictions['probabilidad_reposicion'][y_class_test == 1]
    proba_neg = predictions['probabilidad_reposicion'][y_class_test == 0]
    
    plt.hist(proba_neg, bins=30, alpha=0.7, label='No Necesita', color='blue')
    plt.hist(proba_pos, bins=30, alpha=0.7, label='Necesita', color='red')
    plt.axvline(predictor.threshold, color='black', linestyle='--', 
                label=f'Umbral ({predictor.threshold:.3f})')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades - Modelo Optimizado')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{plots_dir}/probability_distribution_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizaciones guardadas")

def save_final_predictor(predictor, metrics):
    """Guardar predictor final"""
    print("\n💾 GUARDANDO PREDICTOR OPTIMIZADO")
    print("-" * 40)
    
    # Guardar predictor
    dump(predictor, f'{final_dir}/stock_predictor_optimized.joblib')
    
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None:
            return None
        else:
            return obj
    
    config = {
        'version': '2.0_optimized',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_used': {
            'classifier': 'GradientBoostingClassifier (fresh trained)',
            'regressor': 'LightGBMRegressor (fresh trained)'
        },
        'approach': 'Fresh training with winning configuration from Script 3',
        'performance': clean_for_json(metrics),
        'threshold': float(predictor.threshold),
        'features_count': len(predictor.feature_names),
        'usage': {
            'load': "predictor = joblib.load('models/final/stock_predictor_optimized.joblib')",
            'predict': "predictions = predictor.predict(X_new)"
        }
    }
    
    with open(f'{final_dir}/config_optimized.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Predictor guardado: {final_dir}/stock_predictor_optimized.joblib")
    print(f"✅ Config guardado: {final_dir}/config_optimized.json")

def demo_predictor(predictor):
    """Demo del predictor"""
    print("\n🎮 DEMO DEL PREDICTOR OPTIMIZADO")
    print("-" * 40)
    
    df = pd.read_csv('data/processed/02_features/features_engineered.csv')
    sample = df.sample(n=5, random_state=42)
    
    X_sample = sample[predictor.feature_names]
    predictions = predictor.predict(X_sample)
    
    print("📋 EJEMPLOS:")
    for i, (idx, row) in enumerate(sample.iterrows()):
        real_necesita = bool(row['necesita_reposicion'])
        real_cantidad = float(row['cantidad_a_reponer'])
        
        pred_necesita = bool(predictions['necesita_reposicion'][i])
        pred_proba = float(predictions['probabilidad_reposicion'][i])
        pred_cantidad = float(predictions['cantidad_a_reponer'][i])
        
        print(f"\n   Ejemplo {i+1}:")
        print(f"   • Real: {'SÍ' if real_necesita else 'NO'} ({real_cantidad:.1f} unidades)")
        print(f"   • Predicción: {'SÍ' if pred_necesita else 'NO'} (prob: {pred_proba:.3f})")
        if pred_necesita:
            print(f"   • Cantidad: {pred_cantidad:.1f} unidades")
        
        accuracy = "✅" if pred_necesita == real_necesita else "❌"
        print(f"   • Resultado: {accuracy}")

def main():
    print("🚀 PREDICTOR OPTIMIZADO CON RE-ENTRENAMIENTO")
    print("="*55)
    print("💡 ESTRATEGIA: Entrenar modelos frescos con configuración ganadora")
    print("   • Gradient Boosting para clasificación")
    print("   • LightGBM para regresión")
    print("   • Mismo preprocesamiento que Script 3")
    
    # Cargar datos
    X, y_class, y_reg, y_reg_log, feature_names = load_data()
    
    # Crear y entrenar predictor
    predictor = OptimizedStockPredictor()
    training_metrics = predictor.fit(X, y_class, y_reg_log, feature_names)
    
    print(f"\n📊 MÉTRICAS DE ENTRENAMIENTO:")
    print(f"   • Accuracy: {training_metrics['classification']['accuracy']:.4f}")
    print(f"   • F1-Score: {training_metrics['classification']['f1_score']:.4f}")
    print(f"   • F1-Score optimizado: {training_metrics['classification']['optimized_f1']:.4f}")
    print(f"   • R² Regresión: {training_metrics['regression']['r2']:.4f}")
    
    # Evaluar
    eval_metrics, predictions, test_data = evaluate_predictor(predictor, X, y_class, y_reg)
    
    # Visualizaciones
    create_visualizations(predictor, test_data, predictions)
    
    # Guardar
    save_final_predictor(predictor, eval_metrics)
    
    # Demo
    demo_predictor(predictor)
    
    print(f"\n✅ COMPLETADO")
    print(f"📊 MÉTRICAS FINALES:")
    print(f"   • Accuracy: {eval_metrics['classification']['accuracy']:.4f}")
    print(f"   • F1-Score: {eval_metrics['classification']['f1_score']:.4f}")
    if eval_metrics['regression']['r2'] is not None:
        print(f"   • R² Regresión: {eval_metrics['regression']['r2']:.4f}")
    
    print(f"\n🎯 VENTAJAS:")
    print(f"   ✅ Modelos entrenados específicamente para estos datos")
    print(f"   ✅ Configuración probada del Script 3")
    print(f"   ✅ Umbral optimizado para mejor F1-Score")
    print(f"   ✅ Compatible y reproducible")

if __name__ == "__main__":
    main()