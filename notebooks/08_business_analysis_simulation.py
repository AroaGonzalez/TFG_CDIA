# 08_business_analysis_simulation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar la clase del script 07
try:
    # Intentar ejecutar el script completo para tener acceso a la clase
    with open('notebooks/07_final_models_predictions.py', 'r', encoding='utf-8') as f:
        script_content = f.read()
    exec(script_content)
    # Ahora OptimizedStockPredictor está disponible
    print("✅ Clase OptimizedStockPredictor importada desde script 07")
except Exception as e:
    print(f"⚠️ Error al importar desde script 07: {e}")
    # Si no funciona, intentar importación alternativa
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("final_models", "notebooks/07_final_models_predictions.py")
        final_models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(final_models_module)
        OptimizedStockPredictor = final_models_module.OptimizedStockPredictor
        print("✅ Clase OptimizedStockPredictor importada vía importlib")
    except Exception as e2:
        print(f"❌ Error en importación alternativa: {e2}")
        print("   Definiendo clase localmente como fallback...")
        
        # Fallback: definir clase mínima localmente
        class OptimizedStockPredictor:
            def __init__(self):
                self.classifier = None
                self.regressor = None
                self.scaler = None
                self.reg_scaler = None
                self.feature_names = None
                self.threshold = 0.5
                self.is_trained = False
            
            def predict(self, X):
                # Implementación mínima para compatibilidad
                if not self.is_trained:
                    raise ValueError("El modelo no ha sido entrenado")
                
                if hasattr(X, 'values'):
                    X_array = X.values
                else:
                    X_array = np.array(X)
                
                X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                y_class_proba = self.classifier.predict_proba(X_clean)[:, 1]
                y_class_pred = (y_class_proba > self.threshold).astype(int)
                
                y_reg_pred = np.zeros(len(X_clean))
                positive_mask = y_class_pred == 1
                if positive_mask.sum() > 0:
                    X_reg_scaled = self.reg_scaler.transform(X_clean[positive_mask])
                    y_reg_log = self.regressor.predict(X_reg_scaled)
                    y_reg_original = np.expm1(y_reg_log)
                    y_reg_original = np.clip(y_reg_original, 0, 50000)
                    y_reg_pred[positive_mask] = y_reg_original
                
                return {
                    'necesita_reposicion': y_class_pred,
                    'probabilidad_reposicion': y_class_proba,
                    'cantidad_a_reponer': y_reg_pred
                }

# Configuración
output_dir = 'results/08_business_analysis'
plots_dir = f'{output_dir}/plots'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def load_predictor_and_config():
    """Cargar el predictor optimizado final y su configuración"""
    print("\n🔄 CARGANDO PREDICTOR OPTIMIZADO FINAL")
    print("-" * 50)
    
    try:
        # Cargar predictor del script 07
        predictor = load('models/final/stock_predictor_optimized.joblib')
        print("✅ Predictor optimizado cargado correctamente")
        
        # Cargar configuración
        with open('models/final/config_optimized.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuración cargada:")
        print(f"   • Clasificador: {config['models_used']['classifier']}")
        print(f"   • Regresor: {config['models_used']['regressor']}")
        print(f"   • Umbral optimizado: {config['threshold']:.3f}")
        print(f"   • Features: {config['features_count']}")
        
        return predictor, config
        
    except Exception as e:
        print(f"❌ Error al cargar el predictor: {str(e)}")
        print("   Asegúrate de haber ejecutado el script 07 primero")
        return None, None

def load_business_test_data():
    """Cargar datos específicos para análisis de negocio"""
    print("\n📊 PREPARANDO DATOS PARA ANÁLISIS DE NEGOCIO")
    print("-" * 50)
    
    try:
        # Cargar dataset completo
        df = pd.read_csv('data/processed/02_features/features_engineered.csv')
        
        # Usar 30% para análisis de negocio (diferente del 20% usado en entrenamiento)
        train_val, business_test = train_test_split(
            df, test_size=0.3, random_state=123, stratify=df['necesita_reposicion']
        )
        
        print(f"✅ Datos para análisis de negocio: {business_test.shape[0]:,} registros")
        print(f"   • Balance: {business_test['necesita_reposicion'].mean():.1%} necesitan reposición")
        print(f"   • Productos únicos: {business_test['ID_ALIAS'].nunique()}")
        print(f"   • Tiendas únicas: {business_test['ID_LOCALIZACION_COMPRA'].nunique()}")
        
        return business_test
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        return None

def make_business_predictions(predictor, test_data):
    """Realizar predicciones para análisis de negocio"""
    print("\n🎯 REALIZANDO PREDICCIONES DE NEGOCIO")
    print("-" * 50)
    
    try:
        # Preparar features usando los nombres del predictor
        feature_cols = predictor.feature_names
        
        print(f"✅ Usando {len(feature_cols)} features del predictor entrenado")
        
        # Verificar que todas las features están disponibles
        missing_features = [f for f in feature_cols if f not in test_data.columns]
        if missing_features:
            print(f"⚠️ Features faltantes: {len(missing_features)}")
            # Filtrar solo features disponibles
            available_features = [f for f in feature_cols if f in test_data.columns]
            X = test_data[available_features].copy()
            print(f"   Usando {len(available_features)} features disponibles")
        else:
            X = test_data[feature_cols].copy()
        
        print(f"✅ Features preparadas: {X.shape[1]} columnas")
        
        # Realizar predicciones
        predictions = predictor.predict(X)
        
        # Crear DataFrame con resultados
        results_df = test_data[['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                               'necesita_reposicion', 'cantidad_a_reponer']].copy()
        
        results_df['pred_necesita_reposicion'] = predictions['necesita_reposicion']
        results_df['probabilidad_reposicion'] = predictions['probabilidad_reposicion']
        results_df['pred_cantidad_a_reponer'] = predictions['cantidad_a_reponer']
        
        # Agregar columnas de análisis
        results_df['acierto_clasificacion'] = (
            results_df['necesita_reposicion'] == results_df['pred_necesita_reposicion']
        )
        results_df['error_cantidad_abs'] = np.abs(
            results_df['cantidad_a_reponer'] - results_df['pred_cantidad_a_reponer']
        )
        
        print(f"✅ Predicciones completadas para {len(results_df):,} registros")
        print(f"   • Accuracy general: {results_df['acierto_clasificacion'].mean():.1%}")
        print(f"   • Casos predichos para reposición: {results_df['pred_necesita_reposicion'].sum():,}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error en predicciones: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_business_impact_metrics(results_df):
    """Calcular métricas de impacto en el negocio"""
    print("\n💼 CALCULANDO MÉTRICAS DE IMPACTO EN EL NEGOCIO")
    print("-" * 50)
    
    # Crear matriz de confusión empresarial
    true_pos = results_df[
        (results_df['necesita_reposicion'] == 1) & 
        (results_df['pred_necesita_reposicion'] == 1)
    ]
    false_pos = results_df[
        (results_df['necesita_reposicion'] == 0) & 
        (results_df['pred_necesita_reposicion'] == 1)
    ]
    false_neg = results_df[
        (results_df['necesita_reposicion'] == 1) & 
        (results_df['pred_necesita_reposicion'] == 0)
    ]
    true_neg = results_df[
        (results_df['necesita_reposicion'] == 0) & 
        (results_df['pred_necesita_reposicion'] == 0)
    ]
    
    # 1. Métricas de Clasificación
    accuracy = results_df['acierto_clasificacion'].mean()
    precision = len(true_pos) / (len(true_pos) + len(false_pos)) if (len(true_pos) + len(false_pos)) > 0 else 0
    recall = len(true_pos) / (len(true_pos) + len(false_neg)) if (len(true_pos) + len(false_neg)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 2. Métricas de Gestión de Inventario
    stock_out_rate = len(false_neg) / len(results_df)
    overstock_rate = len(false_pos) / len(results_df)
    service_level = 1 - stock_out_rate
    
    # 3. Métricas Operativas
    correct_restock_decisions = len(true_pos)
    correct_no_restock_decisions = len(true_neg)
    total_correct_decisions = correct_restock_decisions + correct_no_restock_decisions
    missed_restock_opportunities = len(false_neg)
    unnecessary_restock_decisions = len(false_pos)
    
    # Unidades involucradas
    total_units_needed = results_df['cantidad_a_reponer'].sum()
    total_units_predicted = results_df['pred_cantidad_a_reponer'].sum()
    units_excess_inventory = false_pos['pred_cantidad_a_reponer'].sum()
    units_missed_restock = false_neg['cantidad_a_reponer'].sum()
    
    # 4. Métricas de Precisión en Cantidades
    mae_true_positives = true_pos['error_cantidad_abs'].mean() if len(true_pos) > 0 else np.nan
    
    # 5. Eficiencia de Inventario
    eficiencia_inventario = total_units_needed / total_units_predicted if total_units_predicted > 0 else np.nan
    
    # 6. Indicadores de Rendimiento Operativo
    decision_accuracy = total_correct_decisions / len(results_df)
    restock_efficiency = correct_restock_decisions / (correct_restock_decisions + unnecessary_restock_decisions) if (correct_restock_decisions + unnecessary_restock_decisions) > 0 else 0
    restock_coverage = correct_restock_decisions / (correct_restock_decisions + missed_restock_opportunities) if (correct_restock_decisions + missed_restock_opportunities) > 0 else 0
    
    # Consolidar métricas
    business_metrics = {
        'classification_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        },
        'inventory_metrics': {
            'service_level': float(service_level),
            'stock_out_rate': float(stock_out_rate),
            'overstock_rate': float(overstock_rate),
            'inventory_efficiency': float(eficiencia_inventario) if not np.isnan(eficiencia_inventario) else None,
            'decision_accuracy': float(decision_accuracy),
            'restock_efficiency': float(restock_efficiency),
            'restock_coverage': float(restock_coverage)
        },
        'operational_impact': {
            'correct_restock_decisions': int(correct_restock_decisions),
            'correct_no_restock_decisions': int(correct_no_restock_decisions),
            'missed_restock_opportunities': int(missed_restock_opportunities),
            'unnecessary_restock_decisions': int(unnecessary_restock_decisions),
            'total_correct_decisions': int(total_correct_decisions),
            'units_excess_inventory': float(units_excess_inventory),
            'units_missed_restock': float(units_missed_restock)
        },
        'prediction_accuracy': {
            'mae_quantity_true_positives': float(mae_true_positives) if not np.isnan(mae_true_positives) else None,
            'total_required_units': float(total_units_needed),
            'total_predicted_units': float(total_units_predicted),
            'prediction_efficiency': float(total_units_needed / total_units_predicted) if total_units_predicted > 0 else None
        },
        'confusion_matrix': {
            'true_positives': len(true_pos),
            'false_positives': len(false_pos),
            'false_negatives': len(false_neg),
            'true_negatives': len(true_neg)
        }
    }
    
    # Mostrar resultados principales
    print(f"🎯 MÉTRICAS DE CLASIFICACIÓN:")
    print(f"   • Accuracy: {accuracy:.1%}")
    print(f"   • Precision: {precision:.1%}")
    print(f"   • Recall: {recall:.1%}")
    print(f"   • F1-Score: {f1_score:.1%}")
    
    print(f"\n📦 MÉTRICAS DE INVENTARIO:")
    print(f"   • Nivel de Servicio: {service_level:.1%}")
    print(f"   • Tasa de Rotura de Stock: {stock_out_rate:.1%}")
    print(f"   • Tasa de Exceso de Stock: {overstock_rate:.1%}")
    print(f"   • Eficiencia de Inventario: {eficiencia_inventario:.1%}" if not np.isnan(eficiencia_inventario) else "   • Eficiencia de Inventario: N/A")
    print(f"   • Precisión en Decisiones: {decision_accuracy:.1%}")
    print(f"   • Eficiencia en Reposiciones: {restock_efficiency:.1%}")
    print(f"   • Cobertura de Reposiciones: {restock_coverage:.1%}")
    
    print(f"\n📊 IMPACTO OPERATIVO:")
    print(f"   • Decisiones Correctas de Reposición: {correct_restock_decisions:,}")
    print(f"   • Decisiones Correctas de No Reposición: {correct_no_restock_decisions:,}")
    print(f"   • Oportunidades Perdidas: {missed_restock_opportunities:,}")
    print(f"   • Reposiciones Innecesarias: {unnecessary_restock_decisions:,}")
    print(f"   • Unidades en Exceso: {units_excess_inventory:,.0f}")
    print(f"   • Unidades Faltantes: {units_missed_restock:,.0f}")
    
    return business_metrics

def perform_threshold_sensitivity_analysis(predictor, test_data):
    """Análisis de sensibilidad del umbral de decisión"""
    print("\n📈 ANÁLISIS DE SENSIBILIDAD DEL UMBRAL")
    print("-" * 50)
    
    try:
        # Preparar datos usando las features del predictor
        feature_cols = predictor.feature_names
        available_features = [f for f in feature_cols if f in test_data.columns]
        X = test_data[available_features].copy()
        
        # Obtener probabilidades base (sin cambiar umbral)
        old_threshold = predictor.threshold
        predictor.threshold = 0.5  # Temporal para obtener probabilidades base
        
        base_predictions = predictor.predict(X)
        probabilidades = base_predictions['probabilidad_reposicion']
        
        # Restaurar umbral original
        predictor.threshold = old_threshold
        
        # Probar diferentes umbrales
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            # Aplicar nuevo umbral
            new_class_pred = (probabilidades > threshold).astype(int)
            
            # Calcular métricas
            y_true = test_data['necesita_reposicion']
            accuracy = (y_true == new_class_pred).mean()
            
            # Matriz de confusión
            tp = ((y_true == 1) & (new_class_pred == 1)).sum()
            fp = ((y_true == 0) & (new_class_pred == 1)).sum()
            fn = ((y_true == 1) & (new_class_pred == 0)).sum()
            tn = ((y_true == 0) & (new_class_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Métricas de negocio
            service_level = 1 - (fn / len(y_true))
            overstock_rate = fp / len(y_true)
            
            # Score operativo
            operational_score = f1 * service_level * (1 - overstock_rate)
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'service_level': service_level,
                'overstock_rate': overstock_rate,
                'operational_score': operational_score,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            })
        
        # Convertir a DataFrame
        threshold_df = pd.DataFrame(threshold_results)
        
        # Encontrar umbrales óptimos
        best_f1_idx = threshold_df['f1_score'].idxmax()
        best_service_idx = threshold_df['service_level'].idxmax()
        best_operational_idx = threshold_df['operational_score'].idxmax()
        
        optimal_thresholds = {
            'best_f1': {
                'threshold': threshold_df.iloc[best_f1_idx]['threshold'],
                'f1_score': threshold_df.iloc[best_f1_idx]['f1_score'],
                'service_level': threshold_df.iloc[best_f1_idx]['service_level']
            },
            'best_service': {
                'threshold': threshold_df.iloc[best_service_idx]['threshold'],
                'f1_score': threshold_df.iloc[best_service_idx]['f1_score'],
                'service_level': threshold_df.iloc[best_service_idx]['service_level']
            },
            'best_operational': {
                'threshold': threshold_df.iloc[best_operational_idx]['threshold'],
                'f1_score': threshold_df.iloc[best_operational_idx]['f1_score'],
                'operational_score': threshold_df.iloc[best_operational_idx]['operational_score']
            }
        }
        
        # Visualización
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1-Score vs Threshold
        axes[0,0].plot(threshold_df['threshold'], threshold_df['f1_score'], 'b-', linewidth=2)
        axes[0,0].axvline(predictor.threshold, color='r', linestyle='--', label=f'Actual ({predictor.threshold:.2f})')
        axes[0,0].set_title('F1-Score vs Umbral de Decisión')
        axes[0,0].set_xlabel('Umbral')
        axes[0,0].set_ylabel('F1-Score')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Service Level vs Threshold
        axes[0,1].plot(threshold_df['threshold'], threshold_df['service_level'], 'g-', linewidth=2)
        axes[0,1].axvline(predictor.threshold, color='r', linestyle='--', label=f'Actual ({predictor.threshold:.2f})')
        axes[0,1].set_title('Nivel de Servicio vs Umbral')
        axes[0,1].set_xlabel('Umbral')
        axes[0,1].set_ylabel('Nivel de Servicio')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Score Operativo vs Threshold
        axes[1,0].plot(threshold_df['threshold'], threshold_df['operational_score'], 'purple', linewidth=2)
        axes[1,0].axvline(predictor.threshold, color='r', linestyle='--', label=f'Actual ({predictor.threshold:.2f})')
        axes[1,0].set_title('Score Operativo vs Umbral')
        axes[1,0].set_xlabel('Umbral')
        axes[1,0].set_ylabel('Score Operativo')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Trade-off Precision vs Recall
        axes[1,1].plot(threshold_df['recall'], threshold_df['precision'], 'orange', linewidth=2)
        axes[1,1].set_title('Precision vs Recall (Trade-off)')
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análisis completado para {len(thresholds)} umbrales")
        print(f"✅ Umbral actual del modelo: {predictor.threshold:.3f}")
        print(f"✅ Mejor umbral para F1: {optimal_thresholds['best_f1']['threshold']:.3f} (F1: {optimal_thresholds['best_f1']['f1_score']:.3f})")
        print(f"✅ Mejor umbral para servicio: {optimal_thresholds['best_service']['threshold']:.3f} (Servicio: {optimal_thresholds['best_service']['service_level']:.1%})")
        print(f"✅ Mejor umbral operativo: {optimal_thresholds['best_operational']['threshold']:.3f} (Score: {optimal_thresholds['best_operational']['operational_score']:.3f})")
        
        return threshold_df, optimal_thresholds
        
    except Exception as e:
        print(f"❌ Error en análisis de umbral: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

def analyze_performance_by_segments(results_df):
    """Análisis de rendimiento por segmentos de productos y tiendas"""
    print("\n🔍 ANÁLISIS DE RENDIMIENTO POR SEGMENTOS")
    print("-" * 50)
    
    # 1. Análisis por ID_ALIAS (productos)
    print("📦 Analizando rendimiento por productos...")
    
    product_analysis = results_df.groupby('ID_ALIAS').agg({
        'acierto_clasificacion': ['mean', 'count'],
        'necesita_reposicion': 'mean',
        'pred_necesita_reposicion': 'mean',
        'error_cantidad_abs': 'mean',
        'cantidad_a_reponer': 'sum',
        'pred_cantidad_a_reponer': 'sum'
    }).round(3)
    
    # Aplanar nombres de columnas
    product_analysis.columns = ['accuracy', 'count', 'real_restock_rate', 
                               'pred_restock_rate', 'avg_quantity_error',
                               'total_real_quantity', 'total_pred_quantity']
    
    # Filtrar productos con suficientes observaciones
    product_analysis_filtered = product_analysis[product_analysis['count'] >= 3].copy()
    
    # Identificar mejores y peores productos
    if len(product_analysis_filtered) > 0:
        top_products = product_analysis_filtered.nlargest(min(10, len(product_analysis_filtered)), 'accuracy')
        bottom_products = product_analysis_filtered.nsmallest(min(10, len(product_analysis_filtered)), 'accuracy')
    else:
        top_products = pd.DataFrame()
        bottom_products = pd.DataFrame()
    
    # 2. Análisis por ID_LOCALIZACION_COMPRA (tiendas)
    print("🏪 Analizando rendimiento por tiendas...")
    
    store_analysis = results_df.groupby('ID_LOCALIZACION_COMPRA').agg({
        'acierto_clasificacion': ['mean', 'count'],
        'necesita_reposicion': 'mean',
        'pred_necesita_reposicion': 'mean',
        'error_cantidad_abs': 'mean',
        'cantidad_a_reponer': 'sum',
        'pred_cantidad_a_reponer': 'sum'
    }).round(3)
    
    # Aplanar nombres de columnas
    store_analysis.columns = ['accuracy', 'count', 'real_restock_rate', 
                             'pred_restock_rate', 'avg_quantity_error',
                             'total_real_quantity', 'total_pred_quantity']
    
    # Filtrar tiendas con suficientes observaciones
    store_analysis_filtered = store_analysis[store_analysis['count'] >= 2].copy()
    
    # Identificar mejores y peores tiendas
    if len(store_analysis_filtered) > 0:
        top_stores = store_analysis_filtered.nlargest(min(10, len(store_analysis_filtered)), 'accuracy')
        bottom_stores = store_analysis_filtered.nsmallest(min(10, len(store_analysis_filtered)), 'accuracy')
    else:
        top_stores = pd.DataFrame()
        bottom_stores = pd.DataFrame()
    
    # 3. Estadísticas
    product_stats = {
        'total_products_analyzed': len(product_analysis_filtered),
        'avg_accuracy': product_analysis_filtered['accuracy'].mean() if len(product_analysis_filtered) > 0 else 0,
        'std_accuracy': product_analysis_filtered['accuracy'].std() if len(product_analysis_filtered) > 0 else 0,
        'best_product_accuracy': product_analysis_filtered['accuracy'].max() if len(product_analysis_filtered) > 0 else 0,
        'worst_product_accuracy': product_analysis_filtered['accuracy'].min() if len(product_analysis_filtered) > 0 else 0
    }
    
    store_stats = {
        'total_stores_analyzed': len(store_analysis_filtered),
        'avg_accuracy': store_analysis_filtered['accuracy'].mean() if len(store_analysis_filtered) > 0 else 0,
        'std_accuracy': store_analysis_filtered['accuracy'].std() if len(store_analysis_filtered) > 0 else 0,
        'best_store_accuracy': store_analysis_filtered['accuracy'].max() if len(store_analysis_filtered) > 0 else 0,
        'worst_store_accuracy': store_analysis_filtered['accuracy'].min() if len(store_analysis_filtered) > 0 else 0
    }
    
    print(f"✅ Productos analizados: {product_stats['total_products_analyzed']}")
    if product_stats['total_products_analyzed'] > 0:
        print(f"   • Accuracy promedio: {product_stats['avg_accuracy']:.1%}")
        print(f"   • Mejor producto: {product_stats['best_product_accuracy']:.1%}")
        print(f"   • Peor producto: {product_stats['worst_product_accuracy']:.1%}")
    
    print(f"✅ Tiendas analizadas: {store_stats['total_stores_analyzed']}")
    if store_stats['total_stores_analyzed'] > 0:
        print(f"   • Accuracy promedio: {store_stats['avg_accuracy']:.1%}")
        print(f"   • Mejor tienda: {store_stats['best_store_accuracy']:.1%}")
        print(f"   • Peor tienda: {store_stats['worst_store_accuracy']:.1%}")
    
    # Guardar análisis
    product_analysis.to_csv(f'{output_dir}/product_performance_analysis.csv')
    store_analysis.to_csv(f'{output_dir}/store_performance_analysis.csv')
    
    segment_results = {
        'product_stats': product_stats,
        'store_stats': store_stats,
        'top_products': top_products.to_dict('index') if len(top_products) > 0 else {},
        'bottom_products': bottom_products.to_dict('index') if len(bottom_products) > 0 else {},
        'top_stores': top_stores.to_dict('index') if len(top_stores) > 0 else {},
        'bottom_stores': bottom_stores.to_dict('index') if len(bottom_stores) > 0 else {}
    }
    
    return segment_results, product_analysis, store_analysis

def create_business_dashboard_visualizations(results_df, business_metrics):
    """Crear visualizaciones principales para dashboard de negocio"""
    print("\n📊 CREANDO VISUALIZACIONES PARA DASHBOARD")
    print("-" * 50)
    
    # Configuración de estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Dashboard principal
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Matriz de confusión
    cm_data = business_metrics['confusion_matrix']
    confusion_matrix_array = np.array([
        [cm_data['true_negatives'], cm_data['false_positives']],
        [cm_data['false_negatives'], cm_data['true_positives']]
    ])
    
    sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: No Reponer', 'Pred: Reponer'],
                yticklabels=['Real: No Reponer', 'Real: Reponer'],
                ax=axes[0,0])
    axes[0,0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    
    # 2. Distribución de probabilidades
    prob_no_restock = results_df[results_df['necesita_reposicion'] == 0]['probabilidad_reposicion']
    prob_restock = results_df[results_df['necesita_reposicion'] == 1]['probabilidad_reposicion']
    
    axes[0,1].hist(prob_no_restock, bins=30, alpha=0.7, label='No Necesita', color='blue', density=True)
    axes[0,1].hist(prob_restock, bins=30, alpha=0.7, label='Necesita', color='red', density=True)
    axes[0,1].set_xlabel('Probabilidad de Reposición')
    axes[0,1].set_ylabel('Densidad')
    axes[0,1].set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Métricas operativas
    operational_data = business_metrics['operational_impact']
    categories = ['Decisiones\nCorrectas (+)', 'Decisiones\nCorrectas (-)', 'Oportunidades\nPerdidas', 'Reposiciones\nInnecesarias']
    values = [operational_data['correct_restock_decisions'], 
              operational_data['correct_no_restock_decisions'],
              operational_data['missed_restock_opportunities'],
              operational_data['unnecessary_restock_decisions']]
    colors = ['green', 'lightgreen', 'red', 'orange']
    
    bars = axes[1,0].bar(categories, values, color=colors, alpha=0.7)
    axes[1,0].set_title('Impacto Operativo por Categoría', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Número de Casos')
    axes[1,0].grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                      f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. KPIs principales
    kpi_labels = ['Accuracy', 'Nivel de\nServicio', 'Precisión', 'Recall']
    kpi_values = [
        business_metrics['classification_metrics']['accuracy'],
        business_metrics['inventory_metrics']['service_level'],
        business_metrics['classification_metrics']['precision'],
        business_metrics['classification_metrics']['recall']
    ]
    
    bars = axes[1,1].bar(kpi_labels, kpi_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'], alpha=0.8)
    axes[1,1].set_title('KPIs Principales', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Valor de Métrica')
    axes[1,1].set_ylim(0, 1)
    axes[1,1].grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, kpi_values):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/business_dashboard_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Dashboard principal creado")

def generate_executive_summary_report(business_metrics, threshold_analysis, segment_results):
    """Generar reporte ejecutivo completo"""
    print("\n📋 GENERANDO REPORTE EJECUTIVO")
    print("-" * 50)
    
    # Crear reporte ejecutivo
    executive_report = {
        'report_metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Business Impact Analysis',
            'model_version': 'Optimized Hybrid Predictor v2.0'
        },
        
        'executive_summary': {
            'overall_performance': {
                'model_accuracy': business_metrics['classification_metrics']['accuracy'],
                'service_level_achieved': business_metrics['inventory_metrics']['service_level'],
                'decision_accuracy': business_metrics['inventory_metrics']['decision_accuracy'],
                'restock_efficiency': business_metrics['inventory_metrics']['restock_efficiency'],
                'recommendation': 'DEPLOY' if business_metrics['classification_metrics']['accuracy'] > 0.60 and business_metrics['inventory_metrics']['service_level'] > 0.75 else 'OPTIMIZE'
            },
            
            'key_findings': [
                f"El modelo alcanza un {business_metrics['classification_metrics']['accuracy']:.1%} de precisión en la clasificación",
                f"Nivel de servicio al cliente del {business_metrics['inventory_metrics']['service_level']:.1%}",
                f"Precisión en decisiones de inventario del {business_metrics['inventory_metrics']['decision_accuracy']:.1%}",
                f"Eficiencia en reposiciones del {business_metrics['inventory_metrics']['restock_efficiency']:.1%}",
                f"Reducción de rotura de stock al {business_metrics['inventory_metrics']['stock_out_rate']:.1%}",
                f"Tasa de exceso de inventario del {business_metrics['inventory_metrics']['overstock_rate']:.1%}"
            ],
            
            'operational_value_proposition': {
                'correct_decisions': business_metrics['operational_impact']['total_correct_decisions'],
                'service_level': business_metrics['inventory_metrics']['service_level'],
                'inventory_efficiency': business_metrics['inventory_metrics']['inventory_efficiency'],
                'restock_coverage': business_metrics['inventory_metrics']['restock_coverage']
            }
        },
        
        'detailed_metrics': business_metrics,
        
        'optimization_recommendations': {
            'threshold_optimization': {
                'current_threshold': 0.25,
                'optimal_for_f1': threshold_analysis.get('best_f1') if threshold_analysis else None,
                'optimal_for_service': threshold_analysis.get('best_service') if threshold_analysis else None,
                'optimal_for_operations': threshold_analysis.get('best_operational') if threshold_analysis else None
            },
            
            'segment_focus_areas': {
                'products_analyzed': segment_results['product_stats']['total_products_analyzed'],
                'stores_analyzed': segment_results['store_stats']['total_stores_analyzed'],
                'improvement_potential': 'Focus on segments with lower accuracy'
            },
            
            'implementation_roadmap': [
                "Fase 1: Desplegar modelo en tiendas piloto",
                "Fase 2: Optimizar umbral según prioridades operativas",
                "Fase 3: Implementar en todas las tiendas con monitoreo",
                "Fase 4: Mejora continua basada en análisis de segmentos"
            ]
        },
        
        'risk_assessment': {
            'operational_risks': [
                "Dependencia de calidad y completitud de datos",
                "Variación de rendimiento entre categorías de productos",
                "Patrones estacionales pueden requerir actualizaciones",
                f"Tasa de rotura de stock del {business_metrics['inventory_metrics']['stock_out_rate']:.1%}",
                f"Tasa de exceso del {business_metrics['inventory_metrics']['overstock_rate']:.1%}"
            ],
            'mitigation_strategies': [
                "Implementar monitoreo robusto de calidad de datos",
                "Configurar seguimiento automático de rendimiento",
                "Establecer procedimientos de respaldo",
                "Crear estrategias específicas por segmento",
                "Monitorear métricas de nivel de servicio continuamente"
            ]
        },
        
        'next_steps': {
            'immediate_actions': [
                "Validar resultados con stakeholders de negocio",
                "Preparar plan de despliegue piloto",
                "Configurar sistemas de monitoreo y alertas",
                "Entrenar personal en nuevo proceso de gestión de inventario"
            ],
            'medium_term_goals': [
                "Implementar framework de pruebas A/B",
                "Desarrollar capacidades de actualización en tiempo real",
                "Integrar con sistemas existentes de gestión de inventario",
                "Optimizar umbrales por categoría de producto"
            ],
            'long_term_vision': [
                "Expandir a pronóstico de demanda y optimización de cadena de suministro",
                "Implementar gestión dinámica de inventario",
                "Desarrollar recomendaciones personalizadas de reposición",
                "Crear sistema automatizado de optimización de inventario"
            ]
        }
    }
    
    # Guardar reporte ejecutivo
    with open(f'{output_dir}/executive_summary_report.json', 'w') as f:
        json.dump(executive_report, f, indent=2)
    
    # Crear versión legible en texto
    with open(f'{output_dir}/executive_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("REPORTE EJECUTIVO - ANÁLISIS DE IMPACTO OPERATIVO\n")
        f.write("="*55 + "\n\n")
        
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-"*20 + "\n")
        f.write(f"• Precisión del modelo: {business_metrics['classification_metrics']['accuracy']:.1%}\n")
        f.write(f"• Nivel de servicio: {business_metrics['inventory_metrics']['service_level']:.1%}\n")
        f.write(f"• Precisión en decisiones: {business_metrics['inventory_metrics']['decision_accuracy']:.1%}\n")
        f.write(f"• Eficiencia en reposiciones: {business_metrics['inventory_metrics']['restock_efficiency']:.1%}\n")
        f.write(f"• Recomendación: {executive_report['executive_summary']['overall_performance']['recommendation']}\n\n")
        
        f.write("HALLAZGOS CLAVE\n")
        f.write("-"*15 + "\n")
        for finding in executive_report['executive_summary']['key_findings']:
            f.write(f"• {finding}\n")
        f.write("\n")
        
        f.write("IMPACTO OPERATIVO\n")
        f.write("-"*17 + "\n")
        f.write(f"• Decisiones correctas totales: {business_metrics['operational_impact']['total_correct_decisions']:,}\n")
        f.write(f"• Oportunidades perdidas: {business_metrics['operational_impact']['missed_restock_opportunities']:,}\n")
        f.write(f"• Reposiciones innecesarias: {business_metrics['operational_impact']['unnecessary_restock_decisions']:,}\n")
        f.write(f"• Unidades en exceso: {business_metrics['operational_impact']['units_excess_inventory']:,.0f}\n")
        f.write(f"• Unidades faltantes: {business_metrics['operational_impact']['units_missed_restock']:,.0f}\n\n")
        
        f.write("RECOMENDACIONES DE OPTIMIZACIÓN\n")
        f.write("-"*35 + "\n")
        for step in executive_report['optimization_recommendations']['implementation_roadmap']:
            f.write(f"• {step}\n")
        f.write("\n")
        
        f.write("PRÓXIMOS PASOS\n")
        f.write("-"*15 + "\n")
        f.write("Acciones Inmediatas:\n")
        for action in executive_report['next_steps']['immediate_actions']:
            f.write(f"  - {action}\n")
        f.write("\nMetas a Medio Plazo:\n")
        for goal in executive_report['next_steps']['medium_term_goals']:
            f.write(f"  - {goal}\n")
    
    print("✅ Reporte ejecutivo generado")
    print(f"   • Formato JSON: executive_summary_report.json")
    print(f"   • Formato texto: executive_summary_report.txt")
    
    return executive_report

def main():
    print("🚀 ANÁLISIS DE IMPACTO OPERATIVO Y SIMULACIÓN - VERSIÓN CORREGIDA")
    print("="*70)
    
    # 1. Cargar predictor y configuración
    predictor, config = load_predictor_and_config()
    if predictor is None:
        print("❌ No se pudo cargar el predictor. Ejecuta primero el script 07.")
        return None
    
    # 2. Preparar datos de prueba para análisis de negocio
    test_data = load_business_test_data()
    if test_data is None:
        return None
    
    # 3. Realizar predicciones de negocio
    results_df = make_business_predictions(predictor, test_data)
    if results_df is None:
        return None
    
    # 4. Calcular métricas de impacto operativo
    business_metrics = calculate_business_impact_metrics(results_df)
    
    # 5. Análisis de sensibilidad del umbral
    threshold_df, optimal_thresholds = perform_threshold_sensitivity_analysis(predictor, test_data)
    
    # 6. Análisis por segmentos
    segment_results, product_analysis, store_analysis = analyze_performance_by_segments(results_df)
    
    # 7. Crear visualizaciones para dashboard
    create_business_dashboard_visualizations(results_df, business_metrics)
    
    # 8. Generar reporte ejecutivo
    executive_report = generate_executive_summary_report(business_metrics, optimal_thresholds, segment_results)
    
    # 9. Guardar todos los resultados
    results_df.to_csv(f'{output_dir}/business_predictions_analysis.csv', index=False)
    if len(threshold_df) > 0:
        threshold_df.to_csv(f'{output_dir}/threshold_sensitivity_analysis.csv', index=False)
    
    with open(f'{output_dir}/business_metrics_complete.json', 'w') as f:
        json.dump(business_metrics, f, indent=2)
    
    if optimal_thresholds:
        with open(f'{output_dir}/optimal_thresholds.json', 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
    
    with open(f'{output_dir}/segment_analysis_complete.json', 'w') as f:
        json.dump(segment_results, f, indent=2)
    
    print("\n✅ ANÁLISIS OPERATIVO COMPLETADO")
    print(f"📁 Archivos generados:")
    print(f"   • {output_dir}/business_predictions_analysis.csv")
    print(f"   • {output_dir}/business_metrics_complete.json")
    print(f"   • {output_dir}/threshold_sensitivity_analysis.csv")
    print(f"   • {output_dir}/optimal_thresholds.json")
    print(f"   • {output_dir}/product_performance_analysis.csv")
    print(f"   • {output_dir}/store_performance_analysis.csv")
    print(f"   • {output_dir}/segment_analysis_complete.json")
    print(f"   • {output_dir}/executive_summary_report.json")
    print(f"   • {output_dir}/executive_summary_report.txt")
    print(f"   • {plots_dir}/business_dashboard_main.png")
    print(f"   • {plots_dir}/threshold_sensitivity_analysis.png")
    
    print(f"\n🎯 CONCLUSIONES CLAVE:")
    print(f"   • Precisión del modelo: {business_metrics['classification_metrics']['accuracy']:.1%}")
    print(f"   • Nivel de servicio: {business_metrics['inventory_metrics']['service_level']:.1%}")
    print(f"   • Decisiones correctas: {business_metrics['operational_impact']['total_correct_decisions']:,}")
    print(f"   • Productos analizados: {segment_results['product_stats']['total_products_analyzed']}")
    print(f"   • Tiendas analizadas: {segment_results['store_stats']['total_stores_analyzed']}")
    
    # Criterio de recomendación
    deploy_criteria = (
        business_metrics['classification_metrics']['accuracy'] > 0.60 and 
        business_metrics['inventory_metrics']['service_level'] > 0.75
    )
    
    recommendation = "✅ RECOMENDACIÓN: DESPLEGAR EL MODELO" if deploy_criteria else "⚠️ RECOMENDACIÓN: OPTIMIZAR ANTES DE DESPLEGAR"
    print(f"\n{recommendation}")
    
    return business_metrics, results_df, threshold_df, segment_results, executive_report

if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            business_metrics, results_df, threshold_df, segment_results, executive_report = result
            print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        else:
            print("\n❌ ANÁLISIS FALLÓ")
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()