# 07_final_models_predictions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import json
from datetime import datetime
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Configuración
output_dir = 'results/07_final_models'
plots_dir = f'{output_dir}/plots'
models_dir = 'models/final'
predictor_dir = 'models/predictor'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(predictor_dir, exist_ok=True)

class HybridStockPredictor:
    """
    Predictor híbrido para gestión de inventario
    Combina clasificación (¿necesita reposición?) con regresión (¿cuánto reponer?)
    """
    
    def __init__(self, classifier=None, regressor=None, scaler=None, threshold=0.5):
        self.classifier = classifier
        self.regressor = regressor
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = None
        
    def fit(self, X, y_class, y_reg_log, feature_names=None):
        """Entrenar ambos modelos"""
        print("🔄 Entrenando predictor híbrido...")
        
        # Guardar nombres de features
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Entrenar clasificador con todos los datos
        self.classifier.fit(X, y_class)
        
        # Para regresión, usar solo casos positivos
        mask_positive = y_class == 1
        X_reg = X[mask_positive]
        y_reg_positive = y_reg_log[mask_positive]
        
        print(f"   • Clasificador entrenado con {X.shape[0]} muestras")
        print(f"   • Regresor entrenado con {X_reg.shape[0]} muestras positivas")
        
        # Entrenar regresor solo con casos positivos
        self.regressor.fit(X_reg, y_reg_positive)
        
        return self
    
    def predict(self, X):
        """Realizar predicciones híbridas"""
        if self.classifier is None or self.regressor is None:
            raise ValueError("Los modelos no han sido entrenados")
        
        # Convertir a numpy si es DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Manejar NaN e infinitos
        if np.isnan(X_array).any() or np.isinf(X_array).any():
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Escalar datos si hay scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_array)
            # Verificar NaN después del escalado
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_scaled = X_array
        
        # Predicciones de clasificación
        y_class_proba = self.classifier.predict_proba(X_scaled)[:, 1]
        y_class_pred = (y_class_proba > self.threshold).astype(int)
        
        # Predicciones de regresión solo para casos positivos
        y_reg_pred = np.zeros(len(X_scaled))
        positive_indices = np.where(y_class_pred == 1)[0]
        
        if len(positive_indices) > 0:
            X_positive = X_scaled[positive_indices]
            y_reg_log_pred = self.regressor.predict(X_positive)
            y_reg_pred[positive_indices] = np.expm1(y_reg_log_pred)  # Volver a escala original
            
            # Asegurar que las predicciones no sean negativas
            y_reg_pred[positive_indices] = np.maximum(0, y_reg_pred[positive_indices])
        
        return {
            'necesita_reposicion': y_class_pred,
            'probabilidad_reposicion': y_class_proba,
            'cantidad_a_reponer': y_reg_pred
        }
    
    def get_feature_importance(self):
        """Obtener importancia de características"""
        importance_data = {}
        
        if hasattr(self.classifier, 'feature_importances_'):
            importance_data['classification'] = {
                'features': self.feature_names,
                'importance': self.classifier.feature_importances_.tolist()
            }
        
        if hasattr(self.regressor, 'feature_importances_'):
            importance_data['regression'] = {
                'features': self.feature_names,
                'importance': self.regressor.feature_importances_.tolist()
            }
        
        return importance_data

def load_and_prepare_data():
    """Cargar y preparar datos para entrenamiento final"""
    print("\n📊 CARGANDO Y PREPARANDO DATOS FINALES")
    print("-" * 50)
    
    # Cargar datos
    df = pd.read_csv('data/processed/02_features/features_engineered.csv')
    
    # Cargar metadata de features
    try:
        with open('results/02_feature_engineering/feature_metadata.json', 'r') as f:
            metadata = json.load(f)
            features = metadata.get('features', [])
    except FileNotFoundError:
        # Fallback: inferir features
        features = [col for col in df.columns 
                   if col not in ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                                'necesita_reposicion', 'cantidad_a_reponer', 
                                'log_cantidad_a_reponer']]
    
    # Preparar features numéricas
    X = df[features].select_dtypes(include=['number']).copy()
    
    # Manejar valores faltantes si los hay
    if X.isna().any().any():
        print(f"🔍 Detectados valores faltantes, aplicando imputación...")
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
    
    # CLAVE: Eliminar columnas con varianza cero (que causarían NaN en el escalado)
    print(f"🔍 Verificando columnas con varianza cero...")
    zero_var_cols = []
    for col in X.columns:
        if X[col].var() == 0 or X[col].std() == 0:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        print(f"⚠️ Eliminando {len(zero_var_cols)} columnas con varianza cero:")
        for col in zero_var_cols:
            unique_vals = X[col].nunique()
            print(f"   • {col}: {unique_vals} valor(es) único(s)")
        
        X = X.drop(columns=zero_var_cols)
        print(f"✅ Columnas eliminadas. Features restantes: {X.shape[1]}")
    
    # Verificar valores infinitos
    inf_cols = X.columns[np.isinf(X).any()]
    if len(inf_cols) > 0:
        print(f"⚠️ Encontradas {len(inf_cols)} columnas con valores infinitos")
        for col in inf_cols:
            X[col] = X[col].replace([np.inf, -np.inf], 0)
        print("   • Valores infinitos reemplazados con 0")
    
    # Targets
    y_class = df['necesita_reposicion'].copy()
    y_reg = df['cantidad_a_reponer'].copy()
    y_reg_log = df['log_cantidad_a_reponer'].copy() if 'log_cantidad_a_reponer' in df.columns else np.log1p(y_reg)
    
    # Verificar que los targets no tengan NaN
    if y_class.isna().any():
        print("⚠️ Target de clasificación contiene NaN, eliminando filas...")
        valid_mask = y_class.notna()
        X = X[valid_mask]
        y_class = y_class[valid_mask]
        y_reg = y_reg[valid_mask]
        y_reg_log = y_reg_log[valid_mask]
    
    # Verificación final
    print(f"✅ Datos preparados y limpios:")
    print(f"   • Features: {X.shape[1]} (numéricas, varianza > 0)")
    print(f"   • Registros: {X.shape[0]:,}")
    print(f"   • Balance clasificación: {y_class.mean():.1%} positivos")
    print(f"   • Casos para regresión: {(y_reg > 0).sum():,}")
    print(f"   • Rango de varianzas: {X.var().min():.6f} - {X.var().max():.2f}")
    
    return X, y_class, y_reg, y_reg_log, X.columns.tolist()

def train_best_models():
    """Entrenar los mejores modelos identificados en el paso 3"""
    print("\n🏆 ENTRENANDO MEJORES MODELOS DEL PASO 3")
    print("-" * 50)
    
    # Cargar datos
    X, y_class, y_reg, y_reg_log, feature_names = load_and_prepare_data()
    
    # Split para evaluación final
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    y_reg_train = y_reg[X_train.index]
    y_reg_test = y_reg[X_test.index]
    y_reg_log_train = y_reg_log[X_train.index]
    y_reg_log_test = y_reg_log[X_test.index]
    
    # Escalar datos
    print("🔧 Escalando datos...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Verificar NaN después del escalado
    if np.isnan(X_train_scaled).any():
        print("⚠️ Detectados NaN en X_train_scaled después del escalado")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        print("   • NaN e infinitos reemplazados con 0")
    
    if np.isnan(X_test_scaled).any():
        print("⚠️ Detectados NaN en X_test_scaled después del escalado")
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        print("   • NaN e infinitos reemplazados con 0")
    
    # Verificación final antes del entrenamiento
    assert not np.isnan(X_train_scaled).any(), "ERROR: NaN en X_train_scaled"
    assert not np.isnan(X_test_scaled).any(), "ERROR: NaN en X_test_scaled"
    assert not np.isinf(X_train_scaled).any(), "ERROR: Infinitos en X_train_scaled"
    assert not np.isinf(X_test_scaled).any(), "ERROR: Infinitos en X_test_scaled"
    
    print("✅ Datos escalados y verificados")
    
    # Mejor clasificador: Gradient Boosting
    print("\n🎯 Entrenando Gradient Boosting Classifier (mejor del paso 3)")
    best_classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    
    # Mejor regresor: XGBoost (Log)
    print("📈 Entrenando XGBoost Regressor (mejor del paso 3)")
    best_regressor = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    # Crear y entrenar predictor híbrido
    predictor = HybridStockPredictor(
        classifier=best_classifier,
        regressor=best_regressor,
        scaler=scaler,
        threshold=0.5  # Comenzar con umbral estándar
    )
    
    predictor.fit(X_train_scaled, y_class_train, y_reg_log_train, feature_names)
    
    # Optimizar umbral de clasificación
    print("\n🔍 Optimizando umbral de clasificación...")
    y_class_proba = best_classifier.predict_proba(X_test_scaled)[:, 1]
    
    #RECOMENDACION SCRIPT 8 BUSINESS, antes estaba a 0.50
    best_threshold = 0.25
    
    # Actualizar umbral en el predictor
    predictor.threshold = best_threshold
    y_pred_optimal = (y_class_proba > best_threshold).astype(int)
    optimal_f1 = f1_score(y_class_test, y_pred_optimal)

    print(f"✅ Umbral aplicado: {best_threshold:.3f}")
    print(f"✅ F1-Score con umbral optimizado: {optimal_f1:.4f}")
    print("✅ Basado en análisis de costo-beneficio del script 08")
    
    return predictor, (X_test, y_class_test, y_reg_test, y_reg_log_test)

def evaluate_final_model(predictor, test_data):
    """Evaluar el modelo final"""
    print("\n📊 EVALUACIÓN DEL MODELO FINAL")
    print("-" * 50)
    
    X_test, y_class_test, y_reg_test, y_reg_log_test = test_data
    
    # Realizar predicciones
    predictions = predictor.predict(X_test)
    
    # Métricas de clasificación
    y_class_pred = predictions['necesita_reposicion']
    
    class_accuracy = accuracy_score(y_class_test, y_class_pred)
    class_precision = precision_score(y_class_test, y_class_pred)
    class_recall = recall_score(y_class_test, y_class_pred)
    class_f1 = f1_score(y_class_test, y_class_pred)
    
    print(f"🎯 MÉTRICAS DE CLASIFICACIÓN:")
    print(f"   • Accuracy: {class_accuracy:.4f}")
    print(f"   • Precision: {class_precision:.4f}")
    print(f"   • Recall: {class_recall:.4f}")
    print(f"   • F1-Score: {class_f1:.4f}")
    print(f"   • Umbral optimizado: {predictor.threshold:.3f}")
    
    # Métricas de regresión (solo para casos verdaderos positivos)
    true_positive_mask = (y_class_test == 1) & (y_class_pred == 1)
    
    if true_positive_mask.sum() > 0:
        y_reg_true_pos = y_reg_test[true_positive_mask]
        y_reg_pred_pos = predictions['cantidad_a_reponer'][true_positive_mask]
        
        reg_mae = mean_absolute_error(y_reg_true_pos, y_reg_pred_pos)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_true_pos, y_reg_pred_pos))
        reg_r2 = r2_score(y_reg_true_pos, y_reg_pred_pos)
        
        print(f"\n📈 MÉTRICAS DE REGRESIÓN (Verdaderos Positivos):")
        print(f"   • MAE: {reg_mae:.2f}")
        print(f"   • RMSE: {reg_rmse:.2f}")
        print(f"   • R²: {reg_r2:.4f}")
        print(f"   • Casos evaluados: {true_positive_mask.sum()}")
    else:
        reg_mae = reg_rmse = reg_r2 = float('nan')
        print(f"\n⚠️ No hay verdaderos positivos para evaluar regresión")
    
    # Crear matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_class_test, y_class_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Reponer', 'Reponer'],
                yticklabels=['No Reponer', 'Reponer'])
    plt.title('Matriz de Confusión - Modelo Final')
    plt.ylabel('Valores Reales')
    plt.xlabel('Predicciones')
    plt.savefig(f'{plots_dir}/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribución de probabilidades
    plt.figure(figsize=(10, 6))
    proba_pos = predictions['probabilidad_reposicion'][y_class_test == 1]
    proba_neg = predictions['probabilidad_reposicion'][y_class_test == 0]
    
    plt.hist(proba_neg, bins=30, alpha=0.7, label='No Necesita Reposición', color='blue')
    plt.hist(proba_pos, bins=30, alpha=0.7, label='Necesita Reposición', color='red')
    plt.axvline(predictor.threshold, color='black', linestyle='--', 
                label=f'Umbral Optimizado ({predictor.threshold:.3f})')
    plt.xlabel('Probabilidad de Reposición')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades de Reposición')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{plots_dir}/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar métricas
    metrics = {
        'classification': {
            'accuracy': float(class_accuracy),
            'precision': float(class_precision),
            'recall': float(class_recall),
            'f1_score': float(class_f1),
            'threshold': float(predictor.threshold)
        },
        'regression': {
            'mae': float(reg_mae) if not np.isnan(reg_mae) else None,
            'rmse': float(reg_rmse) if not np.isnan(reg_rmse) else None,
            'r2': float(reg_r2) if not np.isnan(reg_r2) else None,
            'evaluated_cases': int(true_positive_mask.sum())
        },
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return metrics

def create_feature_importance_analysis(predictor):
    """Crear análisis de importancia de características"""
    print("\n📈 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("-" * 50)
    
    importance_data = predictor.get_feature_importance()
    
    if 'classification' in importance_data and 'regression' in importance_data:
        # Crear DataFrame para clasificación
        class_imp_df = pd.DataFrame({
            'feature': importance_data['classification']['features'],
            'importance': importance_data['classification']['importance']
        }).sort_values('importance', ascending=False)
        
        # Crear DataFrame para regresión
        reg_imp_df = pd.DataFrame({
            'feature': importance_data['regression']['features'],
            'importance': importance_data['regression']['importance']
        }).sort_values('importance', ascending=False)
        
        # Visualizar top 15 características
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Clasificación
        top_15_class = class_imp_df.head(15)
        sns.barplot(x='importance', y='feature', data=top_15_class, ax=ax1)
        ax1.set_title('Top 15 Características - Clasificación\n(Gradient Boosting)')
        ax1.set_xlabel('Importancia')
        
        # Regresión
        top_15_reg = reg_imp_df.head(15)
        sns.barplot(x='importance', y='feature', data=top_15_reg, ax=ax2)
        ax2.set_title('Top 15 Características - Regresión\n(XGBoost)')
        ax2.set_xlabel('Importancia')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/feature_importance_final.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar importancias
        class_imp_df.to_csv(f'{output_dir}/feature_importance_classification_final.csv', index=False)
        reg_imp_df.to_csv(f'{output_dir}/feature_importance_regression_final.csv', index=False)
        
        print("✅ Análisis de importancia completado y guardado")
        
        return {
            'classification': class_imp_df.head(10).to_dict('records'),
            'regression': reg_imp_df.head(10).to_dict('records')
        }
    
    return {}

def demo_predictor(predictor):
    """Demostrar el uso del predictor con ejemplos reales"""
    print("\n🎮 DEMOSTRACIÓN DEL PREDICTOR")
    print("-" * 50)
    
    try:
        # Cargar dataset original
        df = pd.read_csv('data/processed/02_features/features_engineered.csv')
        
        # Seleccionar muestras representativas
        # 2 casos que necesitan reposición y 2 que no
        positive_samples = df[df['necesita_reposicion'] == 1].sample(n=2, random_state=42)
        negative_samples = df[df['necesita_reposicion'] == 0].sample(n=2, random_state=42)
        
        sample_data = pd.concat([positive_samples, negative_samples])
        
        # Preparar features para predicción
        feature_cols = predictor.feature_names
        X_sample = sample_data[feature_cols]
        
        # Realizar predicciones
        predictions = predictor.predict(X_sample)
        
        examples = []
        
        print("\n📋 EJEMPLOS DE PREDICCIÓN:")
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            real_necesita = bool(row['necesita_reposicion'])
            real_cantidad = float(row['cantidad_a_reponer'])
            
            pred_necesita = bool(predictions['necesita_reposicion'][i])
            pred_proba = float(predictions['probabilidad_reposicion'][i])
            pred_cantidad = float(predictions['cantidad_a_reponer'][i])
            
            print(f"\n   Ejemplo {i+1} (ID_ALIAS: {row['ID_ALIAS']}, ID_LOC: {row['ID_LOCALIZACION_COMPRA']}):")
            print(f"   • Real: {'Reponer' if real_necesita else 'No reponer'} ({real_cantidad:.1f} unidades)")
            print(f"   • Predicción: {'Reponer' if pred_necesita else 'No reponer'} (prob: {pred_proba:.3f})")
            if pred_necesita:
                print(f"   • Cantidad predicha: {pred_cantidad:.1f} unidades")
            
            accuracy_symbol = "✅" if pred_necesita == real_necesita else "❌"
            print(f"   • Clasificación: {accuracy_symbol}")
            
            examples.append({
                'id_alias': int(row['ID_ALIAS']),
                'id_localizacion': int(row['ID_LOCALIZACION_COMPRA']),
                'real': {
                    'necesita_reposicion': real_necesita,
                    'cantidad_a_reponer': real_cantidad
                },
                'prediccion': {
                    'necesita_reposicion': pred_necesita,
                    'probabilidad_reposicion': pred_proba,
                    'cantidad_a_reponer': pred_cantidad
                },
                'clasificacion_correcta': pred_necesita == real_necesita
            })
        
        return examples
        
    except Exception as e:
        print(f"❌ Error en la demostración: {str(e)}")
        return []

def save_final_models(predictor, metrics, importance_analysis, examples):
    """Guardar todos los componentes del modelo final"""
    print("\n💾 GUARDANDO MODELO FINAL Y METADATOS")
    print("-" * 50)
    
    try:
        # Guardar predictor completo
        dump(predictor, f'{predictor_dir}/stock_predictor_final.joblib')
        print(f"✅ Predictor guardado en: {predictor_dir}/stock_predictor_final.joblib")
        
        # Guardar componentes individuales
        dump(predictor.classifier, f'{models_dir}/best_classifier.joblib')
        dump(predictor.regressor, f'{models_dir}/best_regressor.joblib')
        dump(predictor.scaler, f'{models_dir}/features_scaler.joblib')
        
        print(f"✅ Componentes individuales guardados en: {models_dir}/")
        
        # Guardar configuración completa
        config = {
            'model_info': {
                'version': "1.0.0",
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'classifier_type': predictor.classifier.__class__.__name__,
                'regressor_type': predictor.regressor.__class__.__name__,
                'scaler_type': predictor.scaler.__class__.__name__,
                'optimized_threshold': predictor.threshold,
                'feature_count': len(predictor.feature_names)
            },
            'performance_metrics': metrics,
            'feature_importance': importance_analysis,
            'demo_examples': examples,
            'usage_instructions': {
                'loading': "Use joblib.load() to load the predictor",
                'prediction': "Call predictor.predict(X) with scaled feature data",
                'input_format': "DataFrame or numpy array with features in correct order",
                'output_format': {
                    'necesita_reposicion': "boolean array",
                    'probabilidad_reposicion': "float array [0,1]",
                    'cantidad_a_reponer': "float array (units to restock)"
                }
            }
        }
        
        with open(f'{predictor_dir}/model_config_final.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuración completa guardada en: {predictor_dir}/model_config_final.json")
        
        # Guardar lista de features para referencia
        with open(f'{predictor_dir}/feature_names.json', 'w') as f:
            json.dump({
                'features': predictor.feature_names,
                'count': len(predictor.feature_names),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"✅ Lista de features guardada en: {predictor_dir}/feature_names.json")
        
    except Exception as e:
        print(f"❌ Error al guardar: {str(e)}")

def main():
    print("🚀 MODELOS FINALES Y PREDICCIONES")
    print("="*60)
    print("Implementando los mejores modelos identificados en el Paso 3:")
    print("• Clasificación: Gradient Boosting (76.6% accuracy)")
    print("• Regresión: XGBoost Log (R² 0.404, MAE 489.66)")
    
    # Entrenar modelos finales
    predictor, test_data = train_best_models()
    
    # Evaluar modelo final
    metrics = evaluate_final_model(predictor, test_data)
    
    # Análisis de importancia de características
    importance_analysis = create_feature_importance_analysis(predictor)
    
    # Demostración con ejemplos
    examples = demo_predictor(predictor)
    
    # Guardar todo
    save_final_models(predictor, metrics, importance_analysis, examples)
    
    print("\n✅ IMPLEMENTACIÓN COMPLETADA")
    print(f"📁 Archivos generados:")
    print(f"   • {predictor_dir}/stock_predictor_final.joblib")
    print(f"   • {predictor_dir}/model_config_final.json")
    print(f"   • {predictor_dir}/feature_names.json")
    print(f"   • {models_dir}/best_classifier.joblib")
    print(f"   • {models_dir}/best_regressor.joblib")
    print(f"   • {models_dir}/features_scaler.joblib")
    print(f"   • {output_dir}/feature_importance_*.csv")
    print(f"   • {plots_dir}/confusion_matrix_final.png")
    print(f"   • {plots_dir}/probability_distribution.png")
    print(f"   • {plots_dir}/feature_importance_final.png")
    
    print(f"\n🎯 MÉTRICAS FINALES:")
    print(f"   • F1-Score: {metrics['classification']['f1_score']:.4f}")
    print(f"   • Accuracy: {metrics['classification']['accuracy']:.4f}")
    if metrics['regression']['mae'] is not None:
        print(f"   • MAE Regresión: {metrics['regression']['mae']:.2f}")
        print(f"   • R² Regresión: {metrics['regression']['r2']:.4f}")
    
    print(f"\n🏆 LISTO PARA PRODUCCIÓN:")
    print(f"   El predictor híbrido está listo para usar en aplicaciones reales")
    print(f"   Carga el modelo con: joblib.load('{predictor_dir}/stock_predictor_final.joblib')")
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = main()