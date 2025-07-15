# 07_final_models_predictions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
os.makedirs(predictor_dir, exist_ok=True)

# Define la clase HybridPredictor fuera de cualquier función
class HybridPredictor:
    def __init__(self, classifier, regressor, threshold, scaler):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        self.scaler = scaler
        self.version = "1.0.0"
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def preprocess(self, data):
        """Preprocesar datos de entrada"""
        # Convertir a DataFrame si es un diccionario
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Aplicar escalado
        if self.scaler is not None:
            try:
                data_scaled = self.scaler.transform(data)
                # Manejar posibles NaN
                if np.isnan(data_scaled).any():
                    data_scaled = np.nan_to_num(data_scaled, nan=0.0)
                return data_scaled
            except Exception as e:
                print(f"⚠️ Error al escalar datos: {str(e)}")
                return data
        return data
    
    def predict(self, data):
        """Realizar predicción híbrida"""
        # Preprocesar datos
        try:
            data_scaled = self.preprocess(data)
            
            # Clasificación: ¿Necesita reposición?
            if hasattr(self.classifier, 'predict_proba'):
                # Si tiene predict_proba, usarlo con umbral
                class_proba = self.classifier.predict_proba(data_scaled)[:, 1]
                needs_reposition = (class_proba >= self.threshold).astype(int)
            else:
                # Si no, usar predict directamente
                needs_reposition = self.classifier.predict(data_scaled)
            
            # Inicializar array para cantidades a reponer
            reposition_amount = np.zeros(len(data_scaled))
            
            # Regresión: ¿Cuánto reponer? (solo para los que necesitan)
            if needs_reposition.sum() > 0:
                # Indices donde se necesita reposición
                reposition_indices = np.where(needs_reposition == 1)[0]
                
                # Predecir cantidades solo para esos índices
                data_reposition = data_scaled[reposition_indices]
                amount_log = self.regressor.predict(data_reposition)
                
                # Convertir de escala logarítmica a original
                amount = np.expm1(amount_log)
                
                # Asignar cantidades a los índices correspondientes
                for i, idx in enumerate(reposition_indices):
                    reposition_amount[idx] = amount[i]
            
            # Preparar resultados
            results = {
                'necesita_reposicion': needs_reposition,
                'cantidad_a_reponer': reposition_amount
            }
            
            return results
        except Exception as e:
            print(f"❌ Error en la predicción: {str(e)}")
            # Devolver resultados vacíos en caso de error
            return {
                'necesita_reposicion': np.zeros(len(data), dtype=int),
                'cantidad_a_reponer': np.zeros(len(data))
            }
    
    def predict_single(self, data_dict):
        """Método conveniente para predicción de un solo registro"""
        try:
            # Convertir diccionario a DataFrame
            df = pd.DataFrame([data_dict])
            
            # Realizar predicción
            results = self.predict(df)
            
            # Devolver resultados para el único registro
            return {
                'necesita_reposicion': bool(results['necesita_reposicion'][0]),
                'cantidad_a_reponer': float(results['cantidad_a_reponer'][0])
            }
        except Exception as e:
            print(f"❌ Error en predicción individual: {str(e)}")
            return {'necesita_reposicion': False, 'cantidad_a_reponer': 0.0}

def load_models():
    """Cargar los modelos existentes"""
    print("\n🔄 CARGANDO MODELOS EXISTENTES")
    print("-" * 50)
    
    # Cargar el clasificador ensemble
    classifier = None
    try:
        classifier = load('models/final/ensemble_classifier.joblib')
        print("✅ Clasificador ensemble cargado correctamente")
    except FileNotFoundError:
        print("❌ ERROR CRÍTICO: No se encontró el archivo del clasificador")
        print("   Verifica la ruta: 'models/final/ensemble_classifier.joblib'")
        return None, None, None, None
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el clasificador: {str(e)}")
        return None, None, None, None
    
    # Cargar el regresor
    regressor = None
    try:
        regressor = load('models/final/regression_model.joblib')
        print("✅ Modelo de regresión cargado correctamente")
    except FileNotFoundError:
        print("❌ ERROR CRÍTICO: No se encontró el archivo del modelo de regresión")
        print("   Verifica la ruta: 'models/final/regression_model.joblib'")
        return None, None, None, None
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el regresor: {str(e)}")
        return None, None, None, None
    
    # Cargar el scaler
    scaler = None
    try:
        scaler = load('models/final/features_scaler.joblib')
        print("✅ Scaler cargado correctamente")
    except FileNotFoundError:
        print("❌ ERROR CRÍTICO: No se encontró el archivo del scaler")
        print("   Verifica la ruta: 'models/final/features_scaler.joblib'")
        return None, None, None, None
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el scaler: {str(e)}")
        return None, None, None, None
    
    # Obtener umbral de configuración
    threshold = 0.48  # Valor por defecto
    try:
        with open('models/final/hybrid_model_results.json', 'r') as f:
            results = json.load(f)
            threshold = results.get('classification', {}).get('threshold', 0.48)
        print(f"✅ Umbral de clasificación cargado: {threshold}")
    except FileNotFoundError:
        print("⚠️ No se encontró el archivo de configuración del modelo híbrido")
        print(f"⚠️ Usando umbral por defecto: {threshold}")
    except Exception as e:
        print(f"⚠️ Error al cargar la configuración: {str(e)}")
        print(f"⚠️ Usando umbral por defecto: {threshold}")
    
    # Verificación final
    if classifier is None or regressor is None or scaler is None:
        print("❌ ERROR CRÍTICO: No se pudieron cargar todos los componentes necesarios")
        return None, None, None, None
    
    print("✅ Todos los modelos cargados correctamente")
    return classifier, regressor, threshold, scaler

def demo_predictor(predictor):
    """Demostrar el uso del predictor híbrido con ejemplos reales"""
    print("\n🎮 DEMOSTRACIÓN DEL PREDICTOR")
    print("-" * 50)
    
    try:
        # Cargar dataset original
        df = pd.read_csv('data/processed/02_features/features_engineered.csv')
        
        # Seleccionar algunas filas de muestra
        sample_rows = df.sample(n=2, random_state=42)
        
        # Preparar datos para predicción (solo mantener columnas numéricas)
        id_cols = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA']
        target_cols = ['necesita_reposicion', 'cantidad_a_reponer', 'log_cantidad_a_reponer']
        
        # Filtrar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in id_cols + target_cols]
        
        # Realizar predicciones sobre datos de muestra
        X_sample = sample_rows[feature_cols]
        predictions = predictor.predict(X_sample)
        
        # Mostrar resultados para las muestras
        for i in range(len(sample_rows)):
            print(f"\n✅ Ejemplo {i+1}: Datos reales (ID_ALIAS: {sample_rows['ID_ALIAS'].iloc[i]}, ID_LOC: {sample_rows['ID_LOCALIZACION_COMPRA'].iloc[i]})")
            print(f"   • ¿Necesita reposición? {'Sí' if predictions['necesita_reposicion'][i] else 'No'}")
            if predictions['necesita_reposicion'][i]:
                print(f"   • Cantidad a reponer: {predictions['cantidad_a_reponer'][i]:.2f} unidades")
        
        # Guardar ejemplos para referencia
        examples = {
            'ejemplo1': {
                'id_alias': int(sample_rows['ID_ALIAS'].iloc[0]),
                'id_localizacion': int(sample_rows['ID_LOCALIZACION_COMPRA'].iloc[0]),
                'prediccion': {
                    'necesita_reposicion': bool(predictions['necesita_reposicion'][0]),
                    'cantidad_a_reponer': float(predictions['cantidad_a_reponer'][0])
                }
            },
            'ejemplo2': {
                'id_alias': int(sample_rows['ID_ALIAS'].iloc[1]),
                'id_localizacion': int(sample_rows['ID_LOCALIZACION_COMPRA'].iloc[1]),
                'prediccion': {
                    'necesita_reposicion': bool(predictions['necesita_reposicion'][1]),
                    'cantidad_a_reponer': float(predictions['cantidad_a_reponer'][1])
                }
            }
        }
        
        with open(f'{output_dir}/examples.json', 'w') as f:
            json.dump(examples, f, indent=2)
        
        return examples
    except Exception as e:
        print(f"❌ Error en la demostración: {str(e)}")
        return {}

def main():
    print("🚀 IMPLEMENTACIÓN DE PREDICTOR HÍBRIDO FINAL")
    print("="*60)
    
    # Cargar modelos y configuración existentes
    classifier, regressor, threshold, scaler = load_models()
    
    if classifier is None or regressor is None or scaler is None:
        print("❌ ERROR CRÍTICO: No se pudieron cargar todos los componentes necesarios")
        print("   La implementación no puede continuar sin los modelos requeridos")
        return None, {}
    
    # Crear predictor híbrido
    predictor = HybridPredictor(classifier, regressor, threshold, scaler)
    print("✅ Predictor híbrido creado correctamente")
    
    # Guardar el predictor
    try:
        dump(predictor, f'{predictor_dir}/stock_predictor.joblib')
        print(f"✅ Predictor guardado en: {predictor_dir}/stock_predictor.joblib")
        
        # Guardar configuración
        config = {
            'version': "1.0.0",
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'threshold': float(threshold),
            'model_type': {
                'classifier': classifier.__class__.__name__,
                'regressor': regressor.__class__.__name__
            }
        }
        
        with open(f'{predictor_dir}/predictor_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuración guardada en: {predictor_dir}/predictor_config.json")
    except Exception as e:
        print(f"❌ ERROR al guardar el predictor: {str(e)}")
        print("   El predictor se ha creado pero no se ha podido guardar")
    
    # Demostrar predictor con ejemplos
    examples = demo_predictor(predictor)
    
    print("\n✅ IMPLEMENTACIÓN COMPLETADA")
    print(f"📁 Archivos generados:")
    print(f"   • {predictor_dir}/stock_predictor.joblib")
    print(f"   • {predictor_dir}/predictor_config.json")
    print(f"   • {output_dir}/examples.json")
    
    return predictor, examples

if __name__ == "__main__":
    predictor, examples = main()