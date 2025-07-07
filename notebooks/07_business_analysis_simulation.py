# notebooks/07_business_analysis_simulation.py
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

def load_predictions_and_data():
   """Cargar predicciones finales y datos originales"""
   try:
       predictions_df = pd.read_csv('results/stock_predictions_final.csv')
       original_df = pd.read_csv('data/processed/features_engineered.csv')
       
       print(f"✅ Predicciones: {len(predictions_df):,} registros")
       print(f"✅ Datos originales: {len(original_df):,} registros")
       
       return predictions_df, original_df
       
   except FileNotFoundError as e:
       print(f"❌ Error: {e}")
       return None, None

def calculate_business_metrics(predictions_df):
  """Calcular métricas de negocio"""
  print("\n💰 CALCULANDO MÉTRICAS DE NEGOCIO")
  print("-" * 30)
  
  metrics = {}
  
  # 1. Análisis de ocupación
  ocupacion_actual = predictions_df['OCUPACION_ACTUAL'].mean()
  ocupacion_predicha = predictions_df['OCUPACION_PREDICHA'].mean()
  
  metrics['ocupacion'] = {
      'actual_promedio': ocupacion_actual,
      'predicha_promedio': ocupacion_predicha,
      'mejora_ocupacion': ocupacion_predicha - ocupacion_actual
  }
  
  # 2. Casos críticos evitados
  casos_criticos_actuales = (predictions_df['OCUPACION_ACTUAL'] < 0.2).sum()
  casos_criticos_predichos = (predictions_df['OCUPACION_PREDICHA'] < 0.2).sum()
  casos_evitados = casos_criticos_actuales - casos_criticos_predichos
  
  metrics['casos_criticos'] = {
      'actuales': casos_criticos_actuales,
      'predichos': casos_criticos_predichos,
      'evitados': casos_evitados
  }
  
  # 3. Eficiencia de reposición
  total_reposiciones = (predictions_df['NECESITA_REPOSICION'] == 1).sum()
  cantidad_total = predictions_df['CANTIDAD_A_REPONER'].sum()
  
  metrics['reposicion'] = {
      'casos_reposicion': total_reposiciones,
      'porcentaje_tiendas': total_reposiciones / len(predictions_df) * 100,
      'cantidad_total': cantidad_total,
      'cantidad_promedio': cantidad_total / total_reposiciones if total_reposiciones > 0 else 0
  }
  
  # 4. Optimización de inventario
  sobrecapacidad_actual = (predictions_df['OCUPACION_ACTUAL'] > 1.0).sum()
  sobrecapacidad_predicha = (predictions_df['OCUPACION_PREDICHA'] > 1.0).sum()
  
  metrics['inventario'] = {
      'sobrecapacidad_actual': sobrecapacidad_actual,
      'sobrecapacidad_predicha': sobrecapacidad_predicha,
      'mejora_sobrecapacidad': sobrecapacidad_actual - sobrecapacidad_predicha
  }
  
  return metrics

def analyze_by_segments(predictions_df, original_df):
   """Análisis por segmentos de negocio"""
   print("\n🏪 ANÁLISIS POR SEGMENTOS")
   print("-" * 30)
   
   # Crear un DataFrame de análisis combinando datos
   analysis_df = predictions_df.copy()
   
   # Crear variables de segmentación a partir de columnas dummy
   # 1. Tamaño de tienda
   if 'tienda_tamaño_Grande' in original_df.columns:
       tienda_columns = ['tienda_tamaño_Grande', 'tienda_tamaño_Mediana', 'tienda_tamaño_Pequeña']
       merge_df = original_df[['ID_ALIAS', 'ID_LOCALIZACION_COMPRA'] + tienda_columns]
       analysis_df = analysis_df.merge(merge_df, on=['ID_ALIAS', 'ID_LOCALIZACION_COMPRA'], how='left')
       
       analysis_df['tienda_tamaño'] = 'No definido'
       analysis_df.loc[analysis_df['tienda_tamaño_Grande'] == 1, 'tienda_tamaño'] = 'Grande'
       analysis_df.loc[analysis_df['tienda_tamaño_Mediana'] == 1, 'tienda_tamaño'] = 'Mediana'
       analysis_df.loc[analysis_df['tienda_tamaño_Pequeña'] == 1, 'tienda_tamaño'] = 'Pequeña'
   
   # 2. Tipo de alias
   if 'alias_tipo_Alto_volumen' in original_df.columns:
       alias_columns = ['alias_tipo_Alto_volumen', 'alias_tipo_Bajo_volumen', 'alias_tipo_Normal', 'alias_tipo_Volatil']
       merge_df = original_df[['ID_ALIAS', 'ID_LOCALIZACION_COMPRA'] + alias_columns]
       analysis_df = analysis_df.merge(merge_df, on=['ID_ALIAS', 'ID_LOCALIZACION_COMPRA'], how='left')
       
       analysis_df['alias_tipo'] = 'No definido'
       analysis_df.loc[analysis_df['alias_tipo_Alto_volumen'] == 1, 'alias_tipo'] = 'Alto_volumen'
       analysis_df.loc[analysis_df['alias_tipo_Bajo_volumen'] == 1, 'alias_tipo'] = 'Bajo_volumen'
       analysis_df.loc[analysis_df['alias_tipo_Normal'] == 1, 'alias_tipo'] = 'Normal'
       analysis_df.loc[analysis_df['alias_tipo_Volatil'] == 1, 'alias_tipo'] = 'Volatil'
   
   # 3. Urgencia de reposición basada en ocupación
   analysis_df['urgencia_reposicion'] = 0  # Normal
   analysis_df.loc[analysis_df['OCUPACION_ACTUAL'] < 0.5, 'urgencia_reposicion'] = 1  # Medio
   analysis_df.loc[analysis_df['OCUPACION_ACTUAL'] < 0.3, 'urgencia_reposicion'] = 2  # Alto
   analysis_df.loc[analysis_df['OCUPACION_ACTUAL'] < 0.1, 'urgencia_reposicion'] = 3  # Crítico
   
   segment_results = {}
   
   # Análisis por tamaño de tienda
   if 'tienda_tamaño' in analysis_df.columns:
       segment_results['tienda_tamaño'] = {}
       for tamaño in analysis_df['tienda_tamaño'].unique():
           if pd.isna(tamaño):
               continue
           mask = analysis_df['tienda_tamaño'] == tamaño
           subset = analysis_df[mask]
           
           segment_results['tienda_tamaño'][tamaño] = {
               'n_tiendas': len(subset),
               'necesita_reposicion_pct': (subset['NECESITA_REPOSICION'] == 1).mean() * 100,
               'cantidad_promedio': subset['CANTIDAD_A_REPONER'].mean(),
               'ocupacion_actual': subset['OCUPACION_ACTUAL'].mean(),
               'ocupacion_predicha': subset['OCUPACION_PREDICHA'].mean()
           }
   
   # Análisis por tipo de alias
   if 'alias_tipo' in analysis_df.columns:
       segment_results['alias_tipo'] = {}
       for tipo in analysis_df['alias_tipo'].unique():
           if pd.isna(tipo):
               continue
           mask = analysis_df['alias_tipo'] == tipo
           subset = analysis_df[mask]
           
           segment_results['alias_tipo'][tipo] = {
               'n_registros': len(subset),
               'necesita_reposicion_pct': (subset['NECESITA_REPOSICION'] == 1).mean() * 100,
               'cantidad_promedio': subset['CANTIDAD_A_REPONER'].mean(),
               'ocupacion_actual': subset['OCUPACION_ACTUAL'].mean(),
               'ocupacion_predicha': subset['OCUPACION_PREDICHA'].mean()
           }
   
   # Análisis por nivel de urgencia
   segment_results['urgencia'] = {}
   urgencia_map = {0: 'Normal', 1: 'Medio', 2: 'Alto', 3: 'Crítico'}
   
   for urgencia, urgencia_name in urgencia_map.items():
       mask = analysis_df['urgencia_reposicion'] == urgencia
       subset = analysis_df[mask]
       
       if len(subset) > 0:
           segment_results['urgencia'][urgencia_name] = {
               'n_registros': len(subset),
               'necesita_reposicion_pct': (subset['NECESITA_REPOSICION'] == 1).mean() * 100,
               'cantidad_promedio': subset['CANTIDAD_A_REPONER'].mean(),
               'acierto_prediccion': ((subset['urgencia_reposicion'] >= 2) == (subset['NECESITA_REPOSICION'] == 1)).mean() * 100
           }
   
   return segment_results

def simulate_implementation_scenarios(predictions_df):
   """Simular diferentes escenarios de implementación"""
   print("\n🎯 SIMULACIÓN DE ESCENARIOS")
   print("-" * 30)
   
   scenarios = {}
   
   # Escenario 1: Implementación conservadora (solo casos muy seguros)
   threshold_conservador = 0.8
   mask_conservador = predictions_df['PROBABILIDAD_REPOSICION'] >= threshold_conservador
   
   scenarios['conservador'] = {
       'threshold': threshold_conservador,
       'casos_implementados': mask_conservador.sum(),
       'porcentaje_casos': mask_conservador.mean() * 100,
       'cantidad_total': predictions_df[mask_conservador]['CANTIDAD_A_REPONER'].sum(),
       'ocupacion_promedio_mejorada': predictions_df[mask_conservador]['OCUPACION_PREDICHA'].mean()
   }
   
   # Escenario 2: Implementación balanceada
   threshold_balanceado = 0.6
   mask_balanceado = predictions_df['PROBABILIDAD_REPOSICION'] >= threshold_balanceado
   
   scenarios['balanceado'] = {
       'threshold': threshold_balanceado,
       'casos_implementados': mask_balanceado.sum(),
       'porcentaje_casos': mask_balanceado.mean() * 100,
       'cantidad_total': predictions_df[mask_balanceado]['CANTIDAD_A_REPONER'].sum(),
       'ocupacion_promedio_mejorada': predictions_df[mask_balanceado]['OCUPACION_PREDICHA'].mean()
   }
   
   # Escenario 3: Implementación agresiva
   threshold_agresivo = 0.4
   mask_agresivo = predictions_df['PROBABILIDAD_REPOSICION'] >= threshold_agresivo
   
   scenarios['agresivo'] = {
       'threshold': threshold_agresivo,
       'casos_implementados': mask_agresivo.sum(),
       'porcentaje_casos': mask_agresivo.mean() * 100,
       'cantidad_total': predictions_df[mask_agresivo]['CANTIDAD_A_REPONER'].sum(),
       'ocupacion_promedio_mejorada': predictions_df[mask_agresivo]['OCUPACION_PREDICHA'].mean()
   }
   
   return scenarios

def identify_success_failure_cases(predictions_df, original_df):
   """Identificar casos de éxito y fallo"""
   print("\n🔍 IDENTIFICANDO CASOS DE ÉXITO Y FALLO")
   print("-" * 30)
   
   # Usamos la predicción como valor verdadero para la evaluación
   # (en un entorno real tendríamos feedback posterior)
   analysis_df = predictions_df.copy()
   
   # Crear una variable de necesidad real basada en ocupación
   analysis_df['necesita_reposicion_real'] = (analysis_df['OCUPACION_ACTUAL'] < 0.3).astype(int)
   
   # Casos de éxito: predicción correcta
   aciertos = (analysis_df['NECESITA_REPOSICION'] == analysis_df['necesita_reposicion_real'])
   
   # Casos problemáticos
   falsos_positivos = (analysis_df['NECESITA_REPOSICION'] == 1) & (analysis_df['necesita_reposicion_real'] == 0)
   falsos_negativos = (analysis_df['NECESITA_REPOSICION'] == 0) & (analysis_df['necesita_reposicion_real'] == 1)
   
   casos_analisis = {
       'total_casos': len(analysis_df),
       'aciertos': aciertos.sum(),
       'accuracy': aciertos.mean() * 100,
       'falsos_positivos': falsos_positivos.sum(),
       'falsos_negativos': falsos_negativos.sum(),
       'precision_estimada': aciertos.sum() / (aciertos.sum() + falsos_positivos.sum()) * 100 if (aciertos.sum() + falsos_positivos.sum()) > 0 else 0
   }
   
   # Top casos de éxito (alta probabilidad y predicción correcta)
   casos_exito = analysis_df[aciertos & (analysis_df['PROBABILIDAD_REPOSICION'] > 0.8)]
   casos_exito_top = casos_exito.nlargest(10, 'PROBABILIDAD_REPOSICION')[
       ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'PROBABILIDAD_REPOSICION', 'CANTIDAD_A_REPONER']
   ]
   
   # Casos problemáticos para revisar
   casos_problema = analysis_df[falsos_positivos | falsos_negativos]
   casos_problema_top = casos_problema.nlargest(10, 'PROBABILIDAD_REPOSICION')[
       ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'PROBABILIDAD_REPOSICION', 'CANTIDAD_A_REPONER', 
        'NECESITA_REPOSICION', 'necesita_reposicion_real']
   ]
   
   return casos_analisis, casos_exito_top, casos_problema_top

def create_business_visualizations(business_metrics, segment_results, scenarios):
   """Crear visualizaciones de análisis de negocio"""
   print("\n📊 CREANDO VISUALIZACIONES DE NEGOCIO")
   print("-" * 30)
   
   fig, axes = plt.subplots(2, 4, figsize=(20, 12))
   
   # 1. Mejora en ocupación
   ocupaciones = ['Actual', 'Predicha']
   valores_ocupacion = [
       business_metrics['ocupacion']['actual_promedio'],
       business_metrics['ocupacion']['predicha_promedio']
   ]
   axes[0,0].bar(ocupaciones, valores_ocupacion, color=['lightcoral', 'lightgreen'], edgecolor='black')
   axes[0,0].set_title('Ocupación Promedio\nActual vs Predicha')
   axes[0,0].set_ylabel('Ocupación (%)')
   axes[0,0].grid(True, alpha=0.3)
   
   # 2. Casos críticos evitados
   casos_data = [
       business_metrics['casos_criticos']['actuales'],
       business_metrics['casos_criticos']['predichos']
   ]
   axes[0,1].bar(['Actual', 'Con Predicción'], casos_data, color=['red', 'orange'], edgecolor='black')
   axes[0,1].set_title('Casos Críticos\n(Ocupación < 20%)')
   axes[0,1].set_ylabel('Número de casos')
   axes[0,1].grid(True, alpha=0.3)
   
   # 3. Casos implementados por escenario
   if scenarios:
       escenarios = list(scenarios.keys())
       casos_implementados = [scenarios[esc]['casos_implementados'] for esc in escenarios]
       axes[0,2].bar(escenarios, casos_implementados, color='skyblue', edgecolor='black')
       axes[0,2].set_title('Casos Implementados\npor Escenario')
       axes[0,2].set_ylabel('Número de casos')
       axes[0,2].tick_params(axis='x', rotation=45)
       axes[0,2].grid(True, alpha=0.3)
   
   # 4. Distribución por tamaño de tienda
   if 'tienda_tamaño' in segment_results:
       tamaños = list(segment_results['tienda_tamaño'].keys())
       reposicion_pct = [segment_results['tienda_tamaño'][t]['necesita_reposicion_pct'] for t in tamaños]
       axes[0,3].bar(tamaños, reposicion_pct, color='purple', edgecolor='black')
       axes[0,3].set_title('% Necesita Reposición\npor Tamaño Tienda')
       axes[0,3].set_ylabel('% Casos')
       axes[0,3].grid(True, alpha=0.3)
   
   # 5. Cantidad a reponer total
   axes[1,0].bar(['Cantidad Total'], [business_metrics['reposicion']['cantidad_total']], color='green', edgecolor='black')
   axes[1,0].set_title('Cantidad Total a Reponer')
   axes[1,0].set_ylabel('Unidades')
   axes[1,0].grid(True, alpha=0.3)
   
   # 6. Cantidad promedio por tipo de alias
   if 'alias_tipo' in segment_results:
       tipos = list(segment_results['alias_tipo'].keys())
       cantidades = [segment_results['alias_tipo'][t]['cantidad_promedio'] for t in tipos]
       axes[1,1].bar(tipos, cantidades, color='brown', edgecolor='black')
       axes[1,1].set_title('Cantidad Promedio\npor Tipo Alias')
       axes[1,1].set_ylabel('Unidades')
       axes[1,1].tick_params(axis='x', rotation=45)
       axes[1,1].grid(True, alpha=0.3)
   
   # 7. Acierto por nivel de urgencia
   if 'urgencia' in segment_results:
       urgencias = list(segment_results['urgencia'].keys())
       aciertos = [segment_results['urgencia'][u]['acierto_prediccion'] for u in urgencias]
       axes[1,2].bar(urgencias, aciertos, color='gold', edgecolor='black')
       axes[1,2].set_title('Acierto Predicción\npor Urgencia')
       axes[1,2].set_ylabel('% Acierto')
       axes[1,2].tick_params(axis='x', rotation=45)
       axes[1,2].grid(True, alpha=0.3)
   
   # 8. Distribución de thresholds
   if scenarios:
       thresholds = [scenarios[esc]['threshold'] for esc in escenarios]
       porcentajes = [scenarios[esc]['porcentaje_casos'] for esc in escenarios]
       axes[1,3].plot(thresholds, porcentajes, 'o-', color='navy', linewidth=2, markersize=8)
       axes[1,3].set_title('% Casos Implementados\nvs Threshold')
       axes[1,3].set_xlabel('Threshold Probabilidad')
       axes[1,3].set_ylabel('% Casos')
       axes[1,3].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('results/plots/business_analysis.png', dpi=300, bbox_inches='tight')
   plt.show()

def generate_business_recommendations(business_metrics, segment_results, scenarios, casos_analisis):
   """Generar recomendaciones de negocio"""
   print("\n📋 GENERANDO RECOMENDACIONES DE NEGOCIO")
   print("-" * 30)
   
   recommendations = []
   
   # 1. Recomendación de threshold
   mejor_escenario = max(scenarios.keys(), 
                        key=lambda x: scenarios[x]['casos_implementados'] / (scenarios[x]['threshold'] + 0.1))
   
   recommendations.append({
       'categoria': 'Implementación',
       'recomendacion': f'Usar threshold {scenarios[mejor_escenario]["threshold"]} para implementación inicial',
       'justificacion': f'Balancea casos implementados ({scenarios[mejor_escenario]["casos_implementados"]:,}) con confianza',
       'impacto': 'Alto'
   })
   
   # 2. Segmentos prioritarios
   if 'tienda_tamaño' in segment_results:
       mejor_segmento = max(segment_results['tienda_tamaño'].keys(),
                          key=lambda x: segment_results['tienda_tamaño'][x]['necesita_reposicion_pct'])
       
       recommendations.append({
           'categoria': 'Segmentación',
           'recomendacion': f'Priorizar implementación en tiendas {mejor_segmento}',
           'justificacion': f'{segment_results["tienda_tamaño"][mejor_segmento]["necesita_reposicion_pct"]:.1f}% necesitan reposición',
           'impacto': 'Medio'
       })
   
   # 3. Mejora de ocupación
   mejora_ocupacion = business_metrics['ocupacion']['mejora_ocupacion']
   recommendations.append({
       'categoria': 'Eficiencia',
       'recomendacion': f'Implementar sistema para mejorar ocupación promedio',
       'justificacion': f'Mejora de ocupación estimada: {mejora_ocupacion:.1%}',
       'impacto': 'Alto'
   })
   
   # 4. Calidad del modelo
   if casos_analisis['accuracy'] > 80:
       recommendations.append({
           'categoria': 'Modelo',
           'recomendacion': 'Accuracy del modelo es aceptable para producción',
           'justificacion': f'Accuracy: {casos_analisis["accuracy"]:.1f}%',
           'impacto': 'Medio'
       })
   else:
       recommendations.append({
           'categoria': 'Modelo',
           'recomendacion': 'Mejorar modelo antes de implementación amplia',
           'justificacion': f'Accuracy actual: {casos_analisis["accuracy"]:.1f}%',
           'impacto': 'Alto'
       })
   
   # 5. Monitoreo
   recommendations.append({
       'categoria': 'Monitoreo',
       'recomendacion': 'Implementar sistema de feedback para mejorar predicciones',
       'justificacion': 'Los modelos ML mejoran con datos reales de implementación',
       'impacto': 'Medio'
   })
   
   return recommendations

def save_business_analysis_results(business_metrics, segment_results, scenarios, recommendations, casos_exito, casos_problema):
   """Guardar resultados del análisis de negocio"""
   print("\n💾 GUARDANDO ANÁLISIS DE NEGOCIO")
   print("-" * 30)
   
   # Guardar métricas de negocio
   metrics_flat = {}
   for category, metrics in business_metrics.items():
       for metric, value in metrics.items():
           metrics_flat[f'{category}_{metric}'] = value
   
   pd.DataFrame([metrics_flat]).to_csv('results/business_metrics.csv', index=False)
   
   # Guardar recomendaciones
   pd.DataFrame(recommendations).to_csv('results/business_recommendations.csv', index=False)
   
   # Guardar casos de éxito y problema
   casos_exito.to_csv('results/casos_exito_top.csv', index=False)
   casos_problema.to_csv('results/casos_problema_top.csv', index=False)
   
   # Guardar escenarios
   scenarios_df = pd.DataFrame(scenarios).T
   scenarios_df.to_csv('results/implementation_scenarios.csv')
   
   print("✅ Archivos guardados:")
   print("  • results/business_metrics.csv")
   print("  • results/business_recommendations.csv")
   print("  • results/casos_exito_top.csv")
   print("  • results/casos_problema_top.csv")
   print("  • results/implementation_scenarios.csv")

def print_executive_summary(business_metrics, recommendations):
   """Imprimir resumen ejecutivo"""
   print("\n📊 RESUMEN EJECUTIVO")
   print("="*50)
   
   print(f"📈 MEJORAS OPERATIVAS:")
   print(f"  • Mejora ocupación: {business_metrics['ocupacion']['mejora_ocupacion']:.1%}")
   print(f"  • Casos críticos evitados: {business_metrics['casos_criticos']['evitados']:,}")
   print(f"  • Tiendas que necesitan reposición: {business_metrics['reposicion']['porcentaje_tiendas']:.1f}%")
   print(f"  • Cantidad total a reponer: {business_metrics['reposicion']['cantidad_total']:,.0f} unidades")
   
   print(f"\n🎯 RECOMENDACIONES CLAVE:")
   for i, rec in enumerate(recommendations[:3], 1):
       print(f"  {i}. {rec['recomendacion']}")
       print(f"     → {rec['justificacion']}")

def main():
   print("🚀 INICIANDO ANÁLISIS DE NEGOCIO Y SIMULACIÓN")
   print("="*60)
   
   # Cargar datos
   predictions_df, original_df = load_predictions_and_data()
   if predictions_df is None:
       return
   
   # Calcular métricas de negocio
   business_metrics = calculate_business_metrics(predictions_df)
   
   # Análisis por segmentos
   segment_results = analyze_by_segments(predictions_df, original_df)
   
   # Simular escenarios
   scenarios = simulate_implementation_scenarios(predictions_df)
   
   # Identificar casos de éxito/fallo
   casos_analisis, casos_exito, casos_problema = identify_success_failure_cases(predictions_df, original_df)
   
   # Crear visualizaciones
   create_business_visualizations(business_metrics, segment_results, scenarios)
   
   # Generar recomendaciones
   recommendations = generate_business_recommendations(business_metrics, segment_results, scenarios, casos_analisis)
   
   # Guardar resultados
   save_business_analysis_results(business_metrics, segment_results, scenarios, recommendations, casos_exito, casos_problema)
   
   # Resumen ejecutivo
   print_executive_summary(business_metrics, recommendations)
   
   print(f"\n🎯 PRÓXIMO PASO: 08_dashboard_final_report.py")
   
   return business_metrics, segment_results, scenarios, recommendations

if __name__ == "__main__":
   business_metrics, segment_results, scenarios, recommendations = main()