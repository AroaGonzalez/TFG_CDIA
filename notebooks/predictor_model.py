# src/predictor_model.py
import numpy as np
import pandas as pd
from datetime import datetime

class HybridPredictor:
    def __init__(self, classifier, regressor, threshold, scaler, segment_models=None):
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        self.scaler = scaler
        self.segment_models = segment_models or {}
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
                
                # Convertir de escala logarítmica a original con factor de calibración
                amount = np.expm1(amount_log) * 4.5
                
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