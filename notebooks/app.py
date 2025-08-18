from flask import Flask, request, jsonify
import pandas as pd
import os
import logging
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Rutas a archivos
PREDICTIONS_FILE = 'results/08_business_analysis/business_predictions_analysis.csv'
HISTORICAL_FILE = 'data/processed/stock_data_clean.csv'

def _get_prediction_by_id(id_alias, id_localizacion):
    """Función auxiliar para buscar predicciones por ID sin recursividad"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return None, "Archivo de predicciones no encontrado"
            
        df = pd.read_csv(PREDICTIONS_FILE)
        record = df[(df['ID_ALIAS'] == id_alias) & 
                   (df['ID_LOCALIZACION_COMPRA'] == id_localizacion)]
        
        if record.empty:
            return None, "No se encontraron predicciones"
            
        return {
            'necesita_reposicion': bool(record['pred_necesita_reposicion'].values[0]),
            'cantidad_a_reponer': int(round(record['pred_cantidad_a_reponer'].values[0]))
        }, None
        
    except Exception as e:
        logger.error(f"Error en _get_prediction_by_id: {str(e)}")
        return None, str(e)

def _process_historical_match(hist_record, features):
    """Procesar coincidencia histórica"""
    try:
        stock_recuentos = float(features.get('stockRecuentos', 0))
        capacidad_maxima = float(features.get('capacidadMaxima', 100))
        stock_minimo = float(features.get('stockMinimo', 0))
        
        necesita_reposicion = hist_record['necesita_reposicion'] == 1
        
        if necesita_reposicion:
            # Calcular cantidad basada en patrón histórico
            hist_capacidad = float(hist_record.get('CAPACIDAD_MAXIMA', 100))
            hist_cantidad = float(hist_record['cantidad_a_reponer'])
            
            if hist_capacidad > 0:
                ratio_llenado = hist_cantidad / hist_capacidad
                cantidad_base = ratio_llenado * capacidad_maxima
                cantidad_a_reponer = max(0, int(cantidad_base - stock_recuentos))
                
                # No exceder 80% de capacidad
                limite = capacidad_maxima * 0.8
                if stock_recuentos + cantidad_a_reponer > limite:
                    cantidad_a_reponer = max(0, int(limite - stock_recuentos))
            else:
                cantidad_a_reponer = max(0, int(capacidad_maxima * 0.8 - stock_recuentos))
        else:
            cantidad_a_reponer = 0
            
            # Verificar mínimo
            if stock_minimo > 0 and stock_recuentos < stock_minimo:
                necesita_reposicion = True
                cantidad_a_reponer = max(0, int(capacidad_maxima * 0.5 - stock_recuentos))
        
        return {
            'success': True,
            'prediction': {
                'necesita_reposicion': necesita_reposicion,
                'cantidad_a_reponer': cantidad_a_reponer
            },
            'note': 'Predicción basada en datos históricos exactos'
        }
        
    except Exception as e:
        logger.error(f"Error en _process_historical_match: {str(e)}")
        return {'success': False, 'error': str(e)}

def _fallback_prediction(features):
    """Predicción de fallback cuando no hay datos históricos"""
    try:
        stock_recuentos = float(features.get('stockRecuentos', 0))
        capacidad_maxima = float(features.get('capacidadMaxima', 100))
        stock_minimo = float(features.get('stockMinimo', capacidad_maxima * 0.2))
        
        # Lógica simple: reponer si está por debajo del 30% de capacidad
        necesita_reposicion = stock_recuentos < (capacidad_maxima * 0.3)
        
        if necesita_reposicion:
            # Reponer hasta 70% de capacidad
            cantidad_a_reponer = max(0, int(capacidad_maxima * 0.7 - stock_recuentos))
        else:
            cantidad_a_reponer = 0
        
        return {
            'success': True,
            'prediction': {
                'necesita_reposicion': necesita_reposicion,
                'cantidad_a_reponer': cantidad_a_reponer
            },
            'note': 'Predicción de fallback - no se encontraron datos históricos'
        }
        
    except Exception as e:
        logger.error(f"Error en _fallback_prediction: {str(e)}")
        return {'success': False, 'error': str(e)}

def _predict_from_historical_data(features):
    """Predecir basándose en datos históricos"""
    try:
        if not os.path.exists(HISTORICAL_FILE):
            logger.warning(f"Archivo histórico no encontrado: {HISTORICAL_FILE}")
            return _fallback_prediction(features)
            
        df_historical = pd.read_csv(HISTORICAL_FILE)
        
        # Buscar coincidencias exactas
        historical_match = df_historical[
            (df_historical['ID_ALIAS'] == features['ID_ALIAS']) & 
            (df_historical['ID_LOCALIZACION_COMPRA'] == features['ID_LOCALIZACION_COMPRA'])
        ]
        
        if not historical_match.empty:
            logger.info(f"Encontrada coincidencia histórica exacta para ID_ALIAS={features['ID_ALIAS']}, ID_LOCALIZACION_COMPRA={features['ID_LOCALIZACION_COMPRA']}")
            return _process_historical_match(historical_match.iloc[0], features)
        
        # Si no hay coincidencia exacta, buscar por similitud usando predicciones
        if os.path.exists(PREDICTIONS_FILE):
            df_predictions = pd.read_csv(PREDICTIONS_FILE)
            
            # Priorizar por coincidencias parciales
            similar_records = df_predictions.copy()
            similar_records['match_alias'] = similar_records['ID_ALIAS'] == features['ID_ALIAS']
            similar_records['match_location'] = similar_records['ID_LOCALIZACION_COMPRA'] == features['ID_LOCALIZACION_COMPRA']
            similar_records['match_score'] = similar_records['match_alias'].astype(int) + similar_records['match_location'].astype(int)
            similar_records = similar_records.sort_values('match_score', ascending=False)
            
            if not similar_records.empty and similar_records.iloc[0]['match_score'] > 0:
                most_similar = similar_records.iloc[0]
                
                # Adaptar predicción al contexto actual
                necesita_reposicion = bool(most_similar['pred_necesita_reposicion'])
                
                if necesita_reposicion:
                    stock_recuentos = float(features.get('stockRecuentos', 0))
                    capacidad_maxima = float(features.get('capacidadMaxima', 100))
                    
                    # Usar un ratio conservador para casos similares
                    ratio = 0.6  # Reponer al 60% de capacidad para casos similares
                    objetivo_stock = capacidad_maxima * ratio
                    cantidad_a_reponer = max(0, int(objetivo_stock - stock_recuentos))
                else:
                    cantidad_a_reponer = 0
                
                return {
                    'success': True,
                    'prediction': {
                        'necesita_reposicion': necesita_reposicion,
                        'cantidad_a_reponer': cantidad_a_reponer
                    },
                    'note': 'Predicción por similitud con casos conocidos'
                }
        
        # Si no hay nada similar, usar fallback
        return _fallback_prediction(features)
        
    except Exception as e:
        logger.error(f"Error en predicción histórica: {str(e)}")
        return _fallback_prediction(features)


@app.route('/predict_from_id', methods=['POST'])
def predict_from_id():
    """Obtener predicción basada en ID_ALIAS e ID_LOCALIZACION_COMPRA"""
    try:
        data = request.json
        logger.info(f"Recibida solicitud de predicción: {data}")
        
        # Validar datos de entrada
        if not data:
            return jsonify({"success": False, "error": "No se recibieron datos en la solicitud"}), 400
            
        id_alias = data.get('ID_ALIAS')
        id_localizacion = data.get('ID_LOCALIZACION_COMPRA')
        
        # Verificar que se proporcionaron los IDs necesarios
        if id_alias is None or id_localizacion is None:
            return jsonify({
                "success": False, 
                "error": "Se requieren los campos ID_ALIAS e ID_LOCALIZACION_COMPRA"
            }), 400
        
        # Convertir a tipos numéricos si son strings
        try:
            if isinstance(id_alias, str):
                id_alias = int(id_alias)
            if isinstance(id_localizacion, str):
                id_localizacion = int(id_localizacion)
        except ValueError:
            return jsonify({
                "success": False, 
                "error": "Los IDs deben ser valores numéricos"
            }), 400
        
        # Buscar predicción
        prediction, error = _get_prediction_by_id(id_alias, id_localizacion)
        
        if prediction:
            logger.info(f"Predicción encontrada: {prediction}")
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            logger.warning(f"No se encontraron predicciones para ID_ALIAS={id_alias}, ID_LOCALIZACION_COMPRA={id_localizacion}")
            return jsonify({
                "success": False, 
                "error": error or "No se encontraron predicciones para los IDs proporcionados"
            }), 404
    
    except Exception as e:
        logger.error(f"Error inesperado en predict_from_id: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": f"Error inesperado: {str(e)}"
        }), 500

@app.route('/stock/predict_new_stock', methods=['POST'])
def predict_new_stock():
    """Predecir necesidad de restock para nuevos datos"""
    try:
        data = request.json
        logger.info(f"Recibida solicitud para predecir nuevo stock: {data}")
        
        if not data:
            return jsonify({"success": False, "error": "No se recibieron datos en la solicitud"}), 400
            
        features = data.get('features', {})
        
        # Verificar campos requeridos
        required_fields = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'stockRecuentos', 'capacidadMaxima']
        missing_fields = [field for field in required_fields if field not in features]
        
        if missing_fields:
            return jsonify({
                "success": False, 
                "error": f"Faltan campos requeridos: {', '.join(missing_fields)}"
            }), 400
        
        # Intentar con predicción existente primero
        prediction, error = _get_prediction_by_id(
            features['ID_ALIAS'], 
            features['ID_LOCALIZACION_COMPRA']
        )
        
        if prediction:
            # Ajustar cantidad considerando stock actual
            if prediction['necesita_reposicion']:
                stock_recuentos = float(features.get('stockRecuentos', 0))
                capacidad_maxima = float(features.get('capacidadMaxima', 100))
                
                # Objetivo: reponer hasta 80% de capacidad
                objetivo_stock = capacidad_maxima * 0.8
                cantidad_ajustada = max(0, int(objetivo_stock - stock_recuentos))
                prediction['cantidad_a_reponer'] = cantidad_ajustada
                
            return jsonify({
                'success': True,
                'prediction': prediction,
                'note': 'Predicción encontrada y ajustada al stock actual'
            })
        
        # Si no hay predicción exacta, usar datos históricos
        logger.info("No se encontró predicción exacta, buscando en datos históricos...")
        result = _predict_from_historical_data(features)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Error general en predict_new_stock: {str(e)}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": f"Error al predecir stock: {str(e)}"
        }), 500


@app.route('/test', methods=['GET'])
def test():
    """Endpoint simple para verificar que el servicio esté funcionando"""
    return jsonify({
        "status": "working", 
        "service": "stock-predictor-api",
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check completo del servicio y sus dependencias"""
    status = {
        "service": "stock-predictor-api",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Verificar archivos críticos
    files_to_check = {
        "predictions": PREDICTIONS_FILE,
        "historical": HISTORICAL_FILE
    }
    
    file_status = {}
    for name, filepath in files_to_check.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                file_status[name] = {
                    "available": True,
                    "records": len(df),
                    "columns": df.columns.tolist()[:5],  # Solo primeras 5 columnas
                    "last_modified": pd.Timestamp.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).isoformat()
                }
            except Exception as e:
                file_status[name] = {"available": False, "error": str(e)}
                status["status"] = "warning"
        else:
            file_status[name] = {"available": False, "error": "File not found"}
            status["status"] = "warning"
    
    status["data_sources"] = file_status
    return jsonify(status)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas del sistema de predicciones"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({"error": "No hay datos de predicciones disponibles"}), 404
            
        df = pd.read_csv(PREDICTIONS_FILE)
        
        stats = {
            "dataset_info": {
                "total_records": len(df),
                "unique_products": df['ID_ALIAS'].nunique(),
                "unique_locations": df['ID_LOCALIZACION_COMPRA'].nunique()
            },
            "predictions_summary": {
                "need_restock": int(df['pred_necesita_reposicion'].sum()),
                "no_restock": int((df['pred_necesita_reposicion'] == 0).sum()),
                "restock_percentage": f"{(df['pred_necesita_reposicion'].mean() * 100):.1f}%"
            },
            "quantity_stats": {
                "mean_quantity": float(df['pred_cantidad_a_reponer'].mean()),
                "median_quantity": float(df['pred_cantidad_a_reponer'].median()),
                "max_quantity": float(df['pred_cantidad_a_reponer'].max()),
                "total_units": float(df['pred_cantidad_a_reponer'].sum())
            }
        }
        
        # Agregar accuracy si está disponible
        if 'acierto_clasificacion' in df.columns:
            stats["model_performance"] = {
                "accuracy": f"{(df['acierto_clasificacion'].mean() * 100):.1f}%",
                "correct_predictions": int(df['acierto_clasificacion'].sum())
            }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/products', methods=['GET'])
def get_products():
    """Obtener lista de productos disponibles"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({"error": "No hay datos disponibles"}), 404
            
        df = pd.read_csv(PREDICTIONS_FILE)
        products = df['ID_ALIAS'].unique().tolist()
        
        return jsonify({
            "total_products": len(products),
            "products": sorted(products)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/locations', methods=['GET'])
def get_locations():
    """Obtener lista de ubicaciones disponibles"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify({"error": "No hay datos disponibles"}), 404
            
        df = pd.read_csv(PREDICTIONS_FILE)
        locations = df['ID_LOCALIZACION_COMPRA'].unique().tolist()
        
        return jsonify({
            "total_locations": len(locations),
            "locations": sorted(locations)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🚀 INICIANDO STOCK PREDICTOR API")
    logger.info("="*60)
    logger.info(f"📁 Archivo de predicciones: {PREDICTIONS_FILE}")
    logger.info(f"📁 Archivo histórico: {HISTORICAL_FILE}")
    logger.info(f"🌐 Servidor iniciando en puerto 8000...")
    logger.info("="*60)
    
    # Verificar archivos al inicio
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        logger.info(f"✅ Predicciones cargadas: {len(df)} registros")
    else:
        logger.warning(f"⚠️ Archivo de predicciones no encontrado: {PREDICTIONS_FILE}")
    
    if os.path.exists(HISTORICAL_FILE):
        df_hist = pd.read_csv(HISTORICAL_FILE)
        logger.info(f"✅ Datos históricos cargados: {len(df_hist)} registros")
    else:
        logger.warning(f"⚠️ Archivo histórico no encontrado: {HISTORICAL_FILE}")
    
    app.run(host='0.0.0.0', port=8000, debug=True)