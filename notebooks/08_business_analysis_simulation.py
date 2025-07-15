# 08_business_analysis_simulation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import importlib.util
spec = importlib.util.spec_from_file_location("module", os.path.join(os.path.dirname(__file__), "07_final_models_predictions.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HybridPredictor = module.HybridPredictor

# Configuración
output_dir = 'results/08_business_analysis'
plots_dir = f'{output_dir}/plots'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def load_predictor():
   """Cargar el predictor híbrido"""
   try:
       predictor = load('models/predictor/stock_predictor.joblib')
       print("✅ Predictor híbrido cargado correctamente")
       return predictor
   except Exception as e:
       print(f"❌ Error al cargar el predictor: {str(e)}")
       return None

def load_test_data():
   """Cargar datos para simulación y análisis"""
   try:
       # Cargar dataset completo
       df = pd.read_csv('data/processed/02_features/features_engineered.csv')
       
       # Separar 30% para test/simulación
       from sklearn.model_selection import train_test_split
       _, df_test = train_test_split(df, test_size=0.3, random_state=42)
       
       print(f"✅ Datos de prueba cargados: {df_test.shape[0]} registros")
       return df_test
   except Exception as e:
       print(f"❌ Error al cargar datos: {str(e)}")
       return None

def make_predictions(predictor, test_data):
   """Realizar predicciones con el modelo híbrido"""
   try:
       # Preparar datos (solo columnas numéricas)
       id_cols = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA']
       target_cols = ['necesita_reposicion', 'cantidad_a_reponer', 'log_cantidad_a_reponer']
       
       # Filtrar columnas numéricas
       numeric_cols = test_data.select_dtypes(include=['number']).columns.tolist()
       feature_cols = [col for col in numeric_cols if col not in id_cols + target_cols]
       
       # Realizar predicciones
       X_test = test_data[feature_cols]
       predictions = predictor.predict(X_test)
       
       # Crear DataFrame con resultados
       results_df = test_data[id_cols + target_cols].copy()
       results_df['pred_necesita_reposicion'] = predictions['necesita_reposicion']
       results_df['pred_cantidad_a_reponer'] = predictions['cantidad_a_reponer']
       
       print(f"✅ Predicciones realizadas para {len(results_df)} registros")
       return results_df
   except Exception as e:
       print(f"❌ Error al realizar predicciones: {str(e)}")
       return None

def calculate_business_metrics(results_df):
   """Calcular métricas de negocio relevantes"""
   print("\n📊 CALCULANDO MÉTRICAS DE NEGOCIO")
   print("-" * 50)
   
   # Añadir columnas derivadas para análisis
   results_df['acierto_clasificacion'] = results_df['necesita_reposicion'] == results_df['pred_necesita_reposicion']
   results_df['error_cantidad'] = np.abs(results_df['cantidad_a_reponer'] - results_df['pred_cantidad_a_reponer'])
   
   # Calcular casos para análisis
   true_pos = results_df[(results_df['necesita_reposicion'] == 1) & (results_df['pred_necesita_reposicion'] == 1)]
   false_pos = results_df[(results_df['necesita_reposicion'] == 0) & (results_df['pred_necesita_reposicion'] == 1)]
   false_neg = results_df[(results_df['necesita_reposicion'] == 1) & (results_df['pred_necesita_reposicion'] == 0)]
   true_neg = results_df[(results_df['necesita_reposicion'] == 0) & (results_df['pred_necesita_reposicion'] == 0)]
   
   # 1. Exactitud de clasificación
   accuracy = results_df['acierto_clasificacion'].mean()
   
   # 2. Error medio en cantidad (solo para verdaderos positivos)
   mae_verdaderos_positivos = true_pos['error_cantidad'].mean() if len(true_pos) > 0 else np.nan
   
   # 3. Análisis de roturas de stock (falsos negativos)
   tasa_rotura_stock = len(false_neg) / len(results_df)
   
   # 4. Análisis de exceso de stock (falsos positivos)
   tasa_exceso_stock = len(false_pos) / len(results_df)
   
   # 5. Ahorro estimado en stock
   # Suponiendo que el exceso promedio es la diferencia entre predicción y valor real en falsos positivos
   exceso_promedio = false_pos['pred_cantidad_a_reponer'].mean() if len(false_pos) > 0 else 0
   ahorro_stock_unidades = exceso_promedio * len(false_pos)
   
   # 6. Impacto financiero (asumiendo un costo promedio por unidad y costo de oportunidad)
   costo_unitario_promedio = 10  # Valor asumido para simulación
   costo_rotura_stock = 20       # Valor asumido para simulación (costo de oportunidad por venta perdida)
   
   ahorro_financiero = ahorro_stock_unidades * costo_unitario_promedio
   perdida_por_roturas = len(false_neg) * costo_rotura_stock
   balance_financiero = ahorro_financiero - perdida_por_roturas
   
   # 7. Tasa de servicio al cliente (% de casos sin rotura)
   tasa_servicio = 1 - tasa_rotura_stock
   
   # 8. Eficiencia de rotación de inventario
   # Calculada como ratio entre stock real necesario y stock total predicho
   stock_real_necesario = results_df['cantidad_a_reponer'].sum()
   stock_total_predicho = results_df['pred_cantidad_a_reponer'].sum()
   
   if stock_total_predicho > 0:
       eficiencia_rotacion = stock_real_necesario / stock_total_predicho
   else:
       eficiencia_rotacion = np.nan
   
   # Consolidar métricas
   business_metrics = {
       'accuracy': float(accuracy),
       'mae_true_positives': float(mae_verdaderos_positivos),
       'stock_shortage_rate': float(tasa_rotura_stock),
       'excess_stock_rate': float(tasa_exceso_stock),
       'estimated_stock_savings': float(ahorro_stock_unidades),
       'financial_impact': {
           'savings': float(ahorro_financiero),
           'loss_from_shortages': float(perdida_por_roturas),
           'net_balance': float(balance_financiero)
       },
       'customer_service_rate': float(tasa_servicio),
       'inventory_turnover_efficiency': float(eficiencia_rotacion)
   }
   
   # Mostrar resultados
   print(f"✅ Exactitud de clasificación: {accuracy:.2%}")
   print(f"✅ Tasa de servicio al cliente: {tasa_servicio:.2%}")
   print(f"✅ Tasa de rotura de stock: {tasa_rotura_stock:.2%}")
   print(f"✅ Tasa de exceso de stock: {tasa_exceso_stock:.2%}")
   print(f"✅ Ahorro estimado en stock: {ahorro_stock_unidades:.2f} unidades")
   print(f"✅ Impacto financiero neto: {balance_financiero:.2f} unidades monetarias")
   print(f"✅ Eficiencia de rotación de inventario: {eficiencia_rotacion:.2%}")
   
   # Guardar métricas
   with open(f'{output_dir}/business_metrics.json', 'w') as f:
       json.dump(business_metrics, f, indent=2)
   
   return business_metrics, results_df

def perform_scenario_analysis(results_df, business_metrics):
   """Realizar análisis de escenarios variando el umbral de decisión"""
   print("\n📈 ANÁLISIS DE ESCENARIOS")
   print("-" * 50)
   
   # Preparar datos para análisis
   id_cols = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA']
   target_cols = ['necesita_reposicion', 'cantidad_a_reponer']
   
   # Obtener predicciones originales
   base_scenario = {
       'threshold': 0.48,  # Umbral por defecto
       'accuracy': business_metrics['accuracy'],
       'service_rate': business_metrics['customer_service_rate'],
       'stock_savings': business_metrics['estimated_stock_savings'],
       'financial_impact': business_metrics['financial_impact']['net_balance']
   }
   
   # Simular diferentes umbrales
   thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
   scenarios = [base_scenario]
   
   for threshold in thresholds:
       if threshold == 0.48:  # Saltar el umbral base que ya tenemos
           continue
       
       # Recalcular predicciones con nuevo umbral (simulado)
       # En una implementación real habría que volver a correr el modelo,
       # aquí simulamos el cambio de umbral
       
       # Este es un enfoque simplificado para simulación
       # En un entorno real, deberíamos tener las probabilidades y aplicar el umbral
       adjustment_factor = (threshold - 0.48) * 2  # Factor de ajuste simple
       
       # Simular cambio de umbral (reducir positivos si umbral sube, aumentar si baja)
       simulated_df = results_df.copy()
       
       # Simulación simple: cambiamos algunas predicciones según el umbral
       # Esto es solo una aproximación para el análisis de escenarios
       if adjustment_factor > 0:  # Umbral más alto, menos positivos
           # Convertir algunos positivos dudosos a negativos
           mask = (simulated_df['pred_necesita_reposicion'] == 1)
           change_count = int(mask.sum() * adjustment_factor / 2)
           change_indices = simulated_df[mask].sample(n=min(change_count, mask.sum())).index
           simulated_df.loc[change_indices, 'pred_necesita_reposicion'] = 0
           simulated_df.loc[change_indices, 'pred_cantidad_a_reponer'] = 0
       else:  # Umbral más bajo, más positivos
           # Convertir algunos negativos dudosos a positivos
           mask = (simulated_df['pred_necesita_reposicion'] == 0)
           change_count = int(mask.sum() * abs(adjustment_factor) / 2)
           change_indices = simulated_df[mask].sample(n=min(change_count, mask.sum())).index
           simulated_df.loc[change_indices, 'pred_necesita_reposicion'] = 1
           # Asignar una cantidad media a reponer
           simulated_df.loc[change_indices, 'pred_cantidad_a_reponer'] = results_df['pred_cantidad_a_reponer'][results_df['pred_cantidad_a_reponer'] > 0].mean()
       
       # Recalcular métricas para este escenario
       sim_accuracy = (simulated_df['necesita_reposicion'] == simulated_df['pred_necesita_reposicion']).mean()
       
       # Casos para análisis
       true_pos = simulated_df[(simulated_df['necesita_reposicion'] == 1) & (simulated_df['pred_necesita_reposicion'] == 1)]
       false_pos = simulated_df[(simulated_df['necesita_reposicion'] == 0) & (simulated_df['pred_necesita_reposicion'] == 1)]
       false_neg = simulated_df[(simulated_df['necesita_reposicion'] == 1) & (simulated_df['pred_necesita_reposicion'] == 0)]
       
       # Tasa de servicio
       sim_tasa_rotura = len(false_neg) / len(simulated_df)
       sim_tasa_servicio = 1 - sim_tasa_rotura
       
       # Ahorro de stock
       sim_exceso_promedio = false_pos['pred_cantidad_a_reponer'].mean() if len(false_pos) > 0 else 0
       sim_ahorro_stock = sim_exceso_promedio * len(false_pos)
       
       # Impacto financiero
       costo_unitario_promedio = 10
       costo_rotura_stock = 20
       
       sim_ahorro_financiero = sim_ahorro_stock * costo_unitario_promedio
       sim_perdida_roturas = len(false_neg) * costo_rotura_stock
       sim_balance_financiero = sim_ahorro_financiero - sim_perdida_roturas
       
       # Guardar escenario
       scenario = {
           'threshold': threshold,
           'accuracy': float(sim_accuracy),
           'service_rate': float(sim_tasa_servicio),
           'stock_savings': float(sim_ahorro_stock),
           'financial_impact': float(sim_balance_financiero)
       }
       
       scenarios.append(scenario)
   
   # Ordenar escenarios por umbral
   scenarios = sorted(scenarios, key=lambda x: x['threshold'])
   
   # Crear DataFrame para visualización
   scenarios_df = pd.DataFrame(scenarios)
   
   # Visualizar trade-offs
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 2, 1)
   plt.plot(scenarios_df['threshold'], scenarios_df['accuracy'], 'o-', linewidth=2)
   plt.title('Exactitud vs Umbral')
   plt.xlabel('Umbral de Decisión')
   plt.ylabel('Exactitud')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 2)
   plt.plot(scenarios_df['threshold'], scenarios_df['service_rate'], 'o-', linewidth=2)
   plt.title('Tasa de Servicio vs Umbral')
   plt.xlabel('Umbral de Decisión')
   plt.ylabel('Tasa de Servicio')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 3)
   plt.plot(scenarios_df['threshold'], scenarios_df['stock_savings'], 'o-', linewidth=2)
   plt.title('Ahorro de Stock vs Umbral')
   plt.xlabel('Umbral de Decisión')
   plt.ylabel('Unidades Ahorradas')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(2, 2, 4)
   plt.plot(scenarios_df['threshold'], scenarios_df['financial_impact'], 'o-', linewidth=2)
   plt.title('Impacto Financiero vs Umbral')
   plt.xlabel('Umbral de Decisión')
   plt.ylabel('Impacto Financiero Neto')
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig(f'{plots_dir}/scenario_analysis.png', dpi=300)
   
   # Determinar umbral óptimo según criterio financiero
   best_financial_scenario = max(scenarios, key=lambda x: x['financial_impact'])
   
   print(f"\n✅ Análisis de escenarios completado")
   print(f"✅ Umbral óptimo según impacto financiero: {best_financial_scenario['threshold']}")
   print(f"   • Impacto financiero: {best_financial_scenario['financial_impact']:.2f}")
   print(f"   • Tasa de servicio: {best_financial_scenario['service_rate']:.2%}")
   print(f"   • Exactitud: {best_financial_scenario['accuracy']:.2%}")
   
   # Guardar resultados de escenarios
   scenarios_df.to_csv(f'{output_dir}/scenario_analysis.csv', index=False)
   
   # Guardar umbral óptimo
   with open(f'{output_dir}/optimal_threshold.json', 'w') as f:
       json.dump(best_financial_scenario, f, indent=2)
   
   return scenarios_df, best_financial_scenario

def segment_analysis(results_df):
   """Analizar rendimiento del modelo por segmentos (alias y localización)"""
   print("\n🔍 ANÁLISIS POR SEGMENTOS")
   print("-" * 50)
   
   # 1. Análisis por Alias (categoría de producto)
   alias_metrics = results_df.groupby('ID_ALIAS').agg({
       'acierto_clasificacion': 'mean',
       'necesita_reposicion': 'mean',
       'pred_necesita_reposicion': 'mean',
       'error_cantidad': 'mean'
   }).reset_index()
   
   alias_metrics.columns = ['ID_ALIAS', 'Exactitud', 'Tasa_Real_Reposición', 'Tasa_Pred_Reposición', 'Error_Medio_Cantidad']
   
   # Identificar top 5 y bottom 5 alias por exactitud
   top_alias = alias_metrics.nlargest(5, 'Exactitud')
   bottom_alias = alias_metrics.nsmallest(5, 'Exactitud')
   
   # 2. Análisis por Localización (tienda)
   loc_metrics = results_df.groupby('ID_LOCALIZACION_COMPRA').agg({
       'acierto_clasificacion': 'mean',
       'necesita_reposicion': 'mean',
       'pred_necesita_reposicion': 'mean',
       'error_cantidad': 'mean'
   }).reset_index()
   
   loc_metrics.columns = ['ID_LOCALIZACION', 'Exactitud', 'Tasa_Real_Reposición', 'Tasa_Pred_Reposición', 'Error_Medio_Cantidad']
   
   # Identificar top 5 y bottom 5 localizaciones por exactitud
   top_loc = loc_metrics.nlargest(5, 'Exactitud')
   bottom_loc = loc_metrics.nsmallest(5, 'Exactitud')
   
   # Visualización: Top y Bottom Alias
   plt.figure(figsize=(12, 6))
   
   plt.subplot(1, 2, 1)
   sns.barplot(x='ID_ALIAS', y='Exactitud', data=top_alias)
   plt.title('Top 5 Alias por Exactitud')
   plt.ylim(0, 1)
   plt.grid(True, alpha=0.3)
   plt.xticks(rotation=45)
   
   plt.subplot(1, 2, 2)
   sns.barplot(x='ID_ALIAS', y='Exactitud', data=bottom_alias)
   plt.title('Bottom 5 Alias por Exactitud')
   plt.ylim(0, 1)
   plt.grid(True, alpha=0.3)
   plt.xticks(rotation=45)
   
   plt.tight_layout()
   plt.savefig(f'{plots_dir}/alias_performance.png', dpi=300)
   
   # Visualización: Top y Bottom Localizaciones
   plt.figure(figsize=(12, 6))
   
   plt.subplot(1, 2, 1)
   sns.barplot(x='ID_LOCALIZACION', y='Exactitud', data=top_loc)
   plt.title('Top 5 Tiendas por Exactitud')
   plt.ylim(0, 1)
   plt.grid(True, alpha=0.3)
   plt.xticks(rotation=45)
   
   plt.subplot(1, 2, 2)
   sns.barplot(x='ID_LOCALIZACION', y='Exactitud', data=bottom_loc)
   plt.title('Bottom 5 Tiendas por Exactitud')
   plt.ylim(0, 1)
   plt.grid(True, alpha=0.3)
   plt.xticks(rotation=45)
   
   plt.tight_layout()
   plt.savefig(f'{plots_dir}/location_performance.png', dpi=300)
   
   # Guardar resultados
   alias_metrics.to_csv(f'{output_dir}/alias_metrics.csv', index=False)
   loc_metrics.to_csv(f'{output_dir}/location_metrics.csv', index=False)
   
   print(f"✅ Análisis por alias completado: {len(alias_metrics)} alias analizados")
   print(f"✅ Análisis por tienda completado: {len(loc_metrics)} tiendas analizadas")
   
   # Resultados en JSON para dashboard
   segment_results = {
       'top_alias': top_alias[['ID_ALIAS', 'Exactitud']].to_dict('records'),
       'bottom_alias': bottom_alias[['ID_ALIAS', 'Exactitud']].to_dict('records'),
       'top_locations': top_loc[['ID_LOCALIZACION', 'Exactitud']].to_dict('records'),
       'bottom_locations': bottom_loc[['ID_LOCALIZACION', 'Exactitud']].to_dict('records')
   }
   
   with open(f'{output_dir}/segment_analysis.json', 'w') as f:
       json.dump(segment_results, f, indent=2)
   
   return alias_metrics, loc_metrics

def main():
   print("🚀 SIMULACIÓN Y ANÁLISIS DE NEGOCIO")
   print("="*60)
   
   # Cargar predictor híbrido
   predictor = load_predictor()
   if predictor is None:
       return
   
   # Cargar datos de test
   test_data = load_test_data()
   if test_data is None:
       return
   
   # Realizar predicciones
   results_df = make_predictions(predictor, test_data)
   if results_df is None:
       return
   
   # Calcular métricas de negocio
   business_metrics, results_df = calculate_business_metrics(results_df)
   
   # Análisis de escenarios
   scenarios_df, best_scenario = perform_scenario_analysis(results_df, business_metrics)
   
   # Análisis por segmentos
   alias_metrics, loc_metrics = segment_analysis(results_df)
   
   # Guardar dataset con predicciones
   results_df.to_csv(f'{output_dir}/prediction_results.csv', index=False)
   
   print("\n✅ SIMULACIÓN Y ANÁLISIS COMPLETADOS")
   print(f"📁 Archivos generados:")
   print(f"   • {output_dir}/business_metrics.json")
   print(f"   • {output_dir}/scenario_analysis.csv")
   print(f"   • {output_dir}/optimal_threshold.json")
   print(f"   • {output_dir}/alias_metrics.csv")
   print(f"   • {output_dir}/location_metrics.csv")
   print(f"   • {output_dir}/segment_analysis.json")
   print(f"   • {output_dir}/prediction_results.csv")
   print(f"   • {plots_dir}/ (múltiples gráficos)")
   
   return business_metrics, scenarios_df, best_scenario, alias_metrics, loc_metrics

if __name__ == "__main__":
   business_metrics, scenarios_df, best_scenario, alias_metrics, loc_metrics = main()