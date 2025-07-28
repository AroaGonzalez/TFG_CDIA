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

# Ruta al archivo de predicciones
PREDICTIONS_FILE = 'results/08_business_analysis/prediction_results.csv'

@app.route('/predict_from_id', methods=['POST'])
def predict_from_id(data=None):
  try:
      # Usar data si se proporciona, si no, obtener de request
      if data is None:
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
          
      # Verificar que el archivo de predicciones existe
      if not os.path.exists(PREDICTIONS_FILE):
          logger.error(f"Archivo de predicciones no encontrado: {PREDICTIONS_FILE}")
          return jsonify({
              "success": False, 
              "error": "Base de datos de predicciones no disponible"
          }), 500
      
      # Cargar resultados existentes
      try:
          df = pd.read_csv(PREDICTIONS_FILE)
          logger.info(f"Archivo de predicciones cargado: {len(df)} registros")
      except Exception as e:
          logger.error(f"Error al leer archivo de predicciones: {str(e)}")
          return jsonify({
              "success": False, 
              "error": f"Error al leer base de datos de predicciones: {str(e)}"
          }), 500
      
      # Convertir ID_ALIAS e ID_LOCALIZACION_COMPRA a tipos numéricos si son strings
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
      
      # Buscar la predicción correspondiente
      record = df[(df['ID_ALIAS'] == id_alias) & 
                  (df['ID_LOCALIZACION_COMPRA'] == id_localizacion)]
      
      if record.empty:
          logger.warning(f"No se encontraron predicciones para ID_ALIAS={id_alias}, ID_LOCALIZACION_COMPRA={id_localizacion}")
          return jsonify({
              "success": False, 
              "error": "No se encontraron predicciones para los IDs proporcionados"
          }), 404
      
      # Devolver predicción existente
      try:
          response = {
              'success': True,
              'prediction': {
                  'necesita_reposicion': bool(record['pred_necesita_reposicion'].values[0]),
                  'cantidad_a_reponer': int(round(record['pred_cantidad_a_reponer'].values[0]))
              }
          }
          logger.info(f"Predicción encontrada: {response}")
          return jsonify(response)
      except Exception as e:
          logger.error(f"Error al procesar predicción: {str(e)}")
          return jsonify({
              "success": False, 
              "error": f"Error al procesar predicción: {str(e)}"
          }), 500
  
  except Exception as e:
      logger.error(f"Error inesperado: {str(e)}", exc_info=True)
      return jsonify({
          "success": False, 
          "error": f"Error inesperado: {str(e)}"
      }), 500
  
@app.route('/stock/predict_new_stock', methods=['POST'])
def predict_new_stock():
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
     try:
         id_data = {
             'ID_ALIAS': features['ID_ALIAS'],
             'ID_LOCALIZACION_COMPRA': features['ID_LOCALIZACION_COMPRA']
         }
         response = predict_from_id(id_data)
         
         if isinstance(response, tuple):
             prediction_data = response[0]
         else:
             prediction_data = response
         
         if isinstance(prediction_data, dict) and prediction_data.get('success', False):
             # Ajustar la cantidad a reponer considerando stock actual
             if prediction_data.get('prediction', {}).get('necesita_reposicion', False):
                 stock_recuentos = float(features.get('stockRecuentos', 0))
                 capacidad_maxima = float(features.get('capacidadMaxima', 100))
                 
                 # Objetivo: reponer hasta 80% de capacidad
                 objetivo_stock = capacidad_maxima * 0.8
                 cantidad_ajustada = max(0, int(objetivo_stock - stock_recuentos))
                 
                 prediction_data['prediction']['cantidad_a_reponer'] = cantidad_ajustada
                 
             return jsonify(prediction_data)
     except Exception as e:
         logger.warning(f"No se encontró predicción existente: {str(e)}")
     
     # Buscar en datos históricos y predicciones
     try:
         # Cargar ambos datasets
         df_predictions = pd.read_csv(PREDICTIONS_FILE)
         df_historical = pd.read_csv('data/processed/stock_data_clean.csv')
         
         logger.info(f"Datasets cargados: predicciones={len(df_predictions)}, histórico={len(df_historical)}")
         
         # Buscar coincidencias exactas en histórico
         historical_match = df_historical[(df_historical['ID_ALIAS'] == features['ID_ALIAS']) & 
                                       (df_historical['ID_LOCALIZACION_COMPRA'] == features['ID_LOCALIZACION_COMPRA'])]
         
         if not historical_match.empty:
             logger.info(f"Encontrado registro histórico exacto para ID_ALIAS={features['ID_ALIAS']}, ID_LOCALIZACION_COMPRA={features['ID_LOCALIZACION_COMPRA']}")
             
             # Calcular necesidad de reposición
             stock_recuentos = float(features.get('stockRecuentos', 0))
             capacidad_maxima = float(features.get('capacidadMaxima', 100))
             stock_minimo = float(features.get('stockMinimo', 0))
             
             # Usar datos históricos para determinar el patrón de reposición
             hist_record = historical_match.iloc[0]
             necesita_reposicion = hist_record['necesita_reposicion'] == 1
             
             # Si los datos históricos indican que necesita reposición
             if necesita_reposicion:
                 # Adaptar la cantidad histórica proporcionalmente a la capacidad actual
                 hist_capacidad = float(hist_record.get('CAPACIDAD_MAXIMA', 100))
                 hist_cantidad = float(hist_record['cantidad_a_reponer'])
                 
                 if hist_capacidad > 0:
                     # Calcular un ratio de llenado basado en datos históricos
                     ratio_llenado = hist_cantidad / hist_capacidad
                     
                     # Aplicar ratio a capacidad actual
                     cantidad_base = ratio_llenado * capacidad_maxima
                     
                     # Ajustar considerando el stock actual
                     cantidad_a_reponer = max(0, int(cantidad_base - stock_recuentos))
                     
                     # No exceder el 80% de la capacidad
                     limite_superior = capacidad_maxima * 0.8
                     if stock_recuentos + cantidad_a_reponer > limite_superior:
                         cantidad_a_reponer = max(0, int(limite_superior - stock_recuentos))
                 else:
                     # Fallback si no hay capacidad histórica válida
                     objetivo_stock = capacidad_maxima * 0.8
                     cantidad_a_reponer = max(0, int(objetivo_stock - stock_recuentos))
             else:
                 # Si no necesita reposición según datos históricos
                 cantidad_a_reponer = 0
                 
                 # Pero verificar si está por debajo del mínimo actual
                 if stock_minimo > 0 and stock_recuentos < stock_minimo:
                     necesita_reposicion = True
                     cantidad_a_reponer = max(0, int(capacidad_maxima * 0.5 - stock_recuentos))
             
             logger.info(f"Predicción basada en datos históricos: {necesita_reposicion}, {cantidad_a_reponer}")
             
             return jsonify({
                 'success': True,
                 'prediction': {
                     'necesita_reposicion': necesita_reposicion,
                     'cantidad_a_reponer': cantidad_a_reponer
                 },
                 'note': 'Predicción basada en datos históricos'
             })
         
         # Si no hay coincidencia exacta, buscar por similitud
         # Usar búsqueda más flexible
         similar_records = df_predictions.copy()  # Usar todo el dataset
         
         # Ordenar por ID_ALIAS o ID_LOCALIZACION_COMPRA primero
         similar_records['match_alias'] = similar_records['ID_ALIAS'] == features['ID_ALIAS']
         similar_records['match_location'] = similar_records['ID_LOCALIZACION_COMPRA'] == features['ID_LOCALIZACION_COMPRA']
         similar_records['match_score'] = similar_records['match_alias'].astype(int) + similar_records['match_location'].astype(int)
         similar_records = similar_records.sort_values('match_score', ascending=False)
         
         logger.info(f"Registros priorizados por similitud: {len(similar_records)} disponibles")
         
         if not similar_records.empty:
             # Obtener el registro más similar
             most_similar = similar_records.iloc[0]
             
             # Adaptar predicción al contexto actual
             necesita_reposicion = bool(most_similar['pred_necesita_reposicion'])
             
             # Si necesita reposición, calcular la cantidad adaptada
             if necesita_reposicion:
                 stock_recuentos = float(features.get('stockRecuentos', 0))
                 capacidad_maxima = float(features.get('capacidadMaxima', 100))
                 
                 # Calcular un ratio relativo basado en la similitud encontrada
                 # y aplicarlo a la capacidad actual
                 if 'CAPACIDAD_MAXIMA' in most_similar and most_similar['CAPACIDAD_MAXIMA'] > 0:
                     similar_capacidad = float(most_similar['CAPACIDAD_MAXIMA'])
                     ratio = float(most_similar['pred_cantidad_a_reponer']) / similar_capacidad
                 else:
                     # Si no hay datos de capacidad, usar un ratio estándar
                     ratio = 0.7  # Reponer al 70% de capacidad
                 
                 # Aplicar ratio a capacidad actual, considerando stock existente
                 objetivo_stock = capacidad_maxima * ratio
                 cantidad_a_reponer = max(0, int(objetivo_stock - stock_recuentos))
             else:
                 cantidad_a_reponer = 0
             
             logger.info(f"Predicción por similitud adaptada: {necesita_reposicion}, {cantidad_a_reponer}")
             
             return jsonify({
                 'success': True,
                 'prediction': {
                     'necesita_reposicion': necesita_reposicion,
                     'cantidad_a_reponer': cantidad_a_reponer
                 },
                 'note': 'Predicción por similaridad con casos históricos'
             })
         
     except Exception as e:
         logger.error(f"Error en búsqueda histórica: {str(e)}", exc_info=True)
         
 except Exception as e:
     logger.error(f"Error general en predict_new_stock: {str(e)}", exc_info=True)
     return jsonify({
         "success": False, 
         "error": f"Error al predecir stock: {str(e)}"
     }), 500

@app.route('/test', methods=['GET'])
def test():
  return jsonify({"status": "working", "service": "stock-predictor-api"})

@app.route('/health', methods=['GET'])
def health_check():
  """
  Endpoint para verificar que el servicio esté funcionando y tiene acceso a los datos
  """
  status = {
      "service": "stock-predictor-api",
      "status": "healthy",
      "version": "1.0.0"
  }
  
  # Verificar acceso al archivo de predicciones
  if os.path.exists(PREDICTIONS_FILE):
      try:
          df = pd.read_csv(PREDICTIONS_FILE)
          status["data"] = {
              "predictions_file": PREDICTIONS_FILE,
              "record_count": len(df),
              "columns": df.columns.tolist()
          }
      except Exception as e:
          status["status"] = "warning"
          status["data_error"] = str(e)
  else:
      status["status"] = "warning"
      status["data_error"] = f"Archivo de predicciones no encontrado: {PREDICTIONS_FILE}"
  
  return jsonify(status)

if __name__ == '__main__':
  logger.info(f"Iniciando servicio de predicción en puerto 8000...")
  logger.info(f"Archivo de predicciones configurado: {PREDICTIONS_FILE}")
  app.run(host='0.0.0.0', port=8000, debug=True)