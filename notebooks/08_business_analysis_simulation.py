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
    exec(open('notebooks/07_final_models_predictions.py').read())
except:
    # Si no funciona, intentar importación alternativa
    import importlib.util
    spec = importlib.util.spec_from_file_location("final_models", "notebooks/07_final_models_predictions.py")
    final_models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final_models_module)
    HybridStockPredictor = final_models_module.HybridStockPredictor

# Configuración
output_dir = 'results/08_business_analysis'
plots_dir = f'{output_dir}/plots'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def load_predictor_and_config():
    """Cargar el predictor híbrido final y su configuración"""
    print("\n🔄 CARGANDO PREDICTOR HÍBRIDO FINAL")
    print("-" * 50)
    
    try:
        # Cargar predictor
        predictor = load('models/predictor/stock_predictor_final.joblib')
        print("✅ Predictor híbrido cargado correctamente")
        
        # Cargar configuración
        with open('models/predictor/model_config_final.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuración cargada:")
        print(f"   • Modelo: {config['model_info']['classifier_type']} + {config['model_info']['regressor_type']}")
        print(f"   • Umbral optimizado: {config['model_info']['optimized_threshold']:.3f}")
        print(f"   • Features: {config['model_info']['feature_count']}")
        
        return predictor, config
        
    except Exception as e:
        print(f"❌ Error al cargar el predictor: {str(e)}")
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
        # Preparar features (mismo proceso que en entrenamiento)
        features_to_exclude = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                              'necesita_reposicion', 'cantidad_a_reponer', 
                              'log_cantidad_a_reponer']
        
        # Seleccionar features numéricas
        numeric_cols = test_data.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in features_to_exclude]
        
        # Filtrar features que tienen varianza (para evitar problemas con escalado)
        X = test_data[feature_cols].copy()
        
        # Eliminar columnas con varianza cero
        zero_var_cols = []
        for col in X.columns:
            if X[col].var() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            print(f"⚠️ Eliminando {len(zero_var_cols)} columnas sin varianza")
            X = X.drop(columns=zero_var_cols)
        
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
        return None

def calculate_business_impact_metrics(results_df):
    """Calcular métricas de impacto en el negocio (sin estimaciones de costos)"""
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
    
    # Tasa de Rotura de Stock (Stock-out Rate)
    stock_out_rate = len(false_neg) / len(results_df)
    
    # Tasa de Exceso de Stock (Overstock Rate)  
    overstock_rate = len(false_pos) / len(results_df)
    
    # Tasa de Servicio al Cliente (Fill Rate)
    service_level = 1 - stock_out_rate
    
    # 3. Métricas Operativas (sin costos monetarios)
    
    # Número de decisiones correctas e incorrectas
    correct_restock_decisions = len(true_pos)
    correct_no_restock_decisions = len(true_neg)
    total_correct_decisions = correct_restock_decisions + correct_no_restock_decisions
    
    # Casos problemáticos
    missed_restock_opportunities = len(false_neg)  # Roturas de stock
    unnecessary_restock_decisions = len(false_pos)  # Exceso de inventario
    
    # Unidades involucradas
    total_units_needed = results_df['cantidad_a_reponer'].sum()
    total_units_predicted = results_df['pred_cantidad_a_reponer'].sum()
    
    # Unidades en exceso (falsos positivos)
    units_excess_inventory = false_pos['pred_cantidad_a_reponer'].sum()
    
    # Unidades faltantes (falsos negativos)
    units_missed_restock = false_neg['cantidad_a_reponer'].sum()
    
    # 4. Métricas de Precisión en Cantidades
    mae_true_positives = true_pos['error_cantidad_abs'].mean() if len(true_pos) > 0 else np.nan
    
    # 5. Eficiencia de Rotación de Inventario
    eficiencia_inventario = total_units_needed / total_units_predicted if total_units_predicted > 0 else np.nan
    
    # 6. Indicadores de Rendimiento Operativo
    
    # Porcentaje de decisiones acertadas
    decision_accuracy = total_correct_decisions / len(results_df)
    
    # Ratio de eficiencia en reposiciones
    restock_efficiency = correct_restock_decisions / (correct_restock_decisions + unnecessary_restock_decisions) if (correct_restock_decisions + unnecessary_restock_decisions) > 0 else 0
    
    # Cobertura de necesidades de reposición
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
    """Análisis de sensibilidad del umbral de decisión (sin costos monetarios)"""
    print("\n📈 ANÁLISIS DE SENSIBILIDAD DEL UMBRAL")
    print("-" * 50)
    
    # Preparar datos
    features_to_exclude = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                          'necesita_reposicion', 'cantidad_a_reponer', 
                          'log_cantidad_a_reponer']
    
    numeric_cols = test_data.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in features_to_exclude]
    
    X = test_data[feature_cols].copy()
    
    # Eliminar columnas con varianza cero
    for col in X.columns:
        if X[col].var() == 0:
            X = X.drop(columns=[col])
    
    # Obtener probabilidades base
    base_predictions = predictor.predict(X)
    probabilidades = base_predictions['probabilidad_reposicion']
    
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
        
        # Score operativo (combinación de métricas sin dinero)
        # Penalizar tanto roturas como excesos de manera balanceada
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
    
    # Encontrar umbrales óptimos según diferentes criterios
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
    product_analysis_filtered = product_analysis[product_analysis['count'] >= 5].copy()
    
    # Identificar mejores y peores productos
    top_products = product_analysis_filtered.nlargest(10, 'accuracy')
    bottom_products = product_analysis_filtered.nsmallest(10, 'accuracy')
    
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
    
    # Filtrar tiendas con suficientes observaciones (reducir umbral para tiendas)
    store_analysis_filtered = store_analysis[store_analysis['count'] >= 2].copy()
    
    # Identificar mejores y peores tiendas
    top_stores = store_analysis_filtered.nlargest(10, 'accuracy')
    bottom_stores = store_analysis_filtered.nsmallest(10, 'accuracy')
    
    # 3. Visualizaciones
    
    # Distribución de accuracy por productos
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(product_analysis_filtered['accuracy'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribución de Accuracy por Productos')
    plt.xlabel('Accuracy')
    plt.ylabel('Número de Productos')
    plt.grid(True, alpha=0.3)
    
    # Distribución de accuracy por tiendas
    plt.subplot(2, 3, 2)
    plt.hist(store_analysis_filtered['accuracy'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribución de Accuracy por Tiendas')
    plt.xlabel('Accuracy')
    plt.ylabel('Número de Tiendas')
    plt.grid(True, alpha=0.3)
    
    # Top 10 productos
    plt.subplot(2, 3, 3)
    top_10_products = top_products.head(10)
    plt.barh(range(len(top_10_products)), top_10_products['accuracy'])
    plt.yticks(range(len(top_10_products)), [f'Prod {idx}' for idx in top_10_products.index])
    plt.title('Top 10 Productos por Accuracy')
    plt.xlabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Top 10 tiendas
    plt.subplot(2, 3, 4)
    top_10_stores = top_stores.head(10)
    plt.barh(range(len(top_10_stores)), top_10_stores['accuracy'])
    plt.yticks(range(len(top_10_stores)), [f'Tienda {idx}' for idx in top_10_stores.index])
    plt.title('Top 10 Tiendas por Accuracy')
    plt.xlabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Correlación entre volumen y accuracy (productos)
    plt.subplot(2, 3, 5)
    plt.scatter(product_analysis_filtered['count'], product_analysis_filtered['accuracy'], alpha=0.6)
    plt.xlabel('Número de Observaciones')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Volumen (Productos)')
    plt.grid(True, alpha=0.3)
    
    # Correlación entre volumen y accuracy (tiendas)
    plt.subplot(2, 3, 6)
    plt.scatter(store_analysis_filtered['count'], store_analysis_filtered['accuracy'], alpha=0.6)
    plt.xlabel('Número de Observaciones')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Volumen (Tiendas)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/segment_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Análisis de patrones
    
    # Estadísticas descriptivas
    product_stats = {
        'total_products_analyzed': len(product_analysis_filtered),
        'avg_accuracy': product_analysis_filtered['accuracy'].mean(),
        'std_accuracy': product_analysis_filtered['accuracy'].std(),
        'best_product_accuracy': product_analysis_filtered['accuracy'].max(),
        'worst_product_accuracy': product_analysis_filtered['accuracy'].min()
    }
    
    store_stats = {
        'total_stores_analyzed': len(store_analysis_filtered),
        'avg_accuracy': store_analysis_filtered['accuracy'].mean(),
        'std_accuracy': store_analysis_filtered['accuracy'].std(),
        'best_store_accuracy': store_analysis_filtered['accuracy'].max(),
        'worst_store_accuracy': store_analysis_filtered['accuracy'].min()
    }
    
    print(f"✅ Productos analizados: {product_stats['total_products_analyzed']}")
    print(f"   • Accuracy promedio: {product_stats['avg_accuracy']:.1%}")
    print(f"   • Mejor producto: {product_stats['best_product_accuracy']:.1%}")
    print(f"   • Peor producto: {product_stats['worst_product_accuracy']:.1%}")
    
    print(f"✅ Tiendas analizadas: {store_stats['total_stores_analyzed']}")
    print(f"   • Accuracy promedio: {store_stats['avg_accuracy']:.1%}")
    print(f"   • Mejor tienda: {store_stats['best_store_accuracy']:.1%}")
    print(f"   • Peor tienda: {store_stats['worst_store_accuracy']:.1%}")
    
    # Guardar resultados
    segment_results = {
        'product_stats': product_stats,
        'store_stats': store_stats,
        'top_products': top_products.to_dict('index'),
        'bottom_products': bottom_products.to_dict('index'),
        'top_stores': top_stores.to_dict('index'),
        'bottom_stores': bottom_stores.to_dict('index')
    }
    
    # Guardar análisis detallado
    product_analysis.to_csv(f'{output_dir}/product_performance_analysis.csv')
    store_analysis.to_csv(f'{output_dir}/store_performance_analysis.csv')
    
    return segment_results, product_analysis, store_analysis

def create_business_dashboard_visualizations(results_df, business_metrics):
    """Crear visualizaciones principales para dashboard de negocio (sin costos monetarios)"""
    print("\n📊 CREANDO VISUALIZACIONES PARA DASHBOARD DE NEGOCIO")
    print("-" * 50)
    
    # Configuración de estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Dashboard principal (4 gráficos clave)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Matriz de confusión
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
    
    # Distribución de probabilidades
    prob_no_restock = results_df[results_df['necesita_reposicion'] == 0]['probabilidad_reposicion']
    prob_restock = results_df[results_df['necesita_reposicion'] == 1]['probabilidad_reposicion']
    
    axes[0,1].hist(prob_no_restock, bins=30, alpha=0.7, label='No Necesita Reposición', color='blue', density=True)
    axes[0,1].hist(prob_restock, bins=30, alpha=0.7, label='Necesita Reposición', color='red', density=True)
    axes[0,1].axvline(0.3, color='black', linestyle='--', label='Umbral Optimizado')
    axes[0,1].set_xlabel('Probabilidad de Reposición')
    axes[0,1].set_ylabel('Densidad')
    axes[0,1].set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Métricas operativas
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
    
    # Métricas de rendimiento clave
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
    
    # 2. Análisis de errores detallado
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribución de errores en cantidad para verdaderos positivos
    true_positives = results_df[
        (results_df['necesita_reposicion'] == 1) & 
        (results_df['pred_necesita_reposicion'] == 1)
    ]
    
    if len(true_positives) > 0:
        axes[0,0].hist(true_positives['error_cantidad_abs'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0,0].set_xlabel('Error Absoluto en Cantidad')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].set_title('Distribución de Errores en Cantidad\n(Verdaderos Positivos)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Estadísticas del error
        mean_error = true_positives['error_cantidad_abs'].mean()
        median_error = true_positives['error_cantidad_abs'].median()
        axes[0,0].axvline(mean_error, color='red', linestyle='--', label=f'Media: {mean_error:.1f}')
        axes[0,0].axvline(median_error, color='orange', linestyle='--', label=f'Mediana: {median_error:.1f}')
        axes[0,0].legend()
    
    # Scatter plot: Real vs Predicho (cantidades)
    positive_cases = results_df[results_df['pred_necesita_reposicion'] == 1]
    if len(positive_cases) > 0:
        axes[0,1].scatter(positive_cases['cantidad_a_reponer'], 
                         positive_cases['pred_cantidad_a_reponer'], 
                         alpha=0.6, color='blue')
        
        # Línea de referencia (predicción perfecta)
        max_val = max(positive_cases['cantidad_a_reponer'].max(), 
                     positive_cases['pred_cantidad_a_reponer'].max())
        axes[0,1].plot([0, max_val], [0, max_val], 'r--', label='Predicción Perfecta')
        
        axes[0,1].set_xlabel('Cantidad Real a Reponer')
        axes[0,1].set_ylabel('Cantidad Predicha a Reponer')
        axes[0,1].set_title('Cantidad Real vs Predicha\n(Casos Positivos)', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Análisis de unidades involucradas
    unit_categories = ['Unidades\nRequeridas', 'Unidades\nPredichas', 'Unidades en\nExceso', 'Unidades\nFaltantes']
    unit_values = [
        business_metrics['prediction_accuracy']['total_required_units'],
        business_metrics['prediction_accuracy']['total_predicted_units'],
        business_metrics['operational_impact']['units_excess_inventory'],
        business_metrics['operational_impact']['units_missed_restock']
    ]
    
    bars = axes[1,0].bar(unit_categories, unit_values, color=['blue', 'cyan', 'orange', 'red'], alpha=0.7)
    axes[1,0].set_title('Análisis de Unidades', fontweight='bold')
    axes[1,0].set_ylabel('Número de Unidades')
    axes[1,0].grid(True, alpha=0.3)
    
    # Añadir valores
    for bar, value in zip(bars, unit_values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + max(unit_values)*0.01,
                      f'{value:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=45)
    
    # Resumen de tipos de error
    error_types = ['Falsos\nPositivos', 'Falsos\nNegativos']
    error_counts = [cm_data['false_positives'], cm_data['false_negatives']]
    
    bars = axes[1,1].bar(error_types, error_counts, color=['orange', 'red'], alpha=0.7)
    axes[1,1].set_title('Tipos de Errores', fontweight='bold')
    axes[1,1].set_ylabel('Número de Casos')
    axes[1,1].grid(True, alpha=0.3)
    
    # Añadir valores y porcentajes
    total_cases = len(results_df)
    for bar, count in zip(bars, error_counts):
        height = bar.get_height()
        percentage = (count / total_cases) * 100
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(error_counts)*0.02,
                      f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/business_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizaciones del dashboard creadas")

def generate_executive_summary_report(business_metrics, threshold_analysis, segment_results):
    """Generar reporte ejecutivo completo (sin estimaciones de costos)"""
    print("\n📋 GENERANDO REPORTE EJECUTIVO")
    print("-" * 50)
    
    # Crear reporte ejecutivo
    executive_report = {
        'report_metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_type': 'Business Impact Analysis (Operational Focus)',
            'model_version': 'Final Hybrid Predictor v1.0'
        },
        
        'executive_summary': {
            'overall_performance': {
                'model_accuracy': business_metrics['classification_metrics']['accuracy'],
                'service_level_achieved': business_metrics['inventory_metrics']['service_level'],
                'decision_accuracy': business_metrics['inventory_metrics']['decision_accuracy'],
                'restock_efficiency': business_metrics['inventory_metrics']['restock_efficiency'],
                'recommendation': 'DEPLOY' if business_metrics['classification_metrics']['accuracy'] > 0.75 and business_metrics['inventory_metrics']['service_level'] > 0.80 else 'OPTIMIZE'
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
                'current_threshold': 0.3,
                'optimal_for_f1': threshold_analysis.get('best_f1') if threshold_analysis else None,
                'optimal_for_service': threshold_analysis.get('best_service') if threshold_analysis else None,
                'optimal_for_operations': threshold_analysis.get('best_operational') if threshold_analysis else None
            },
            
            'segment_focus_areas': {
                'low_performing_products': len([p for p in segment_results['bottom_products'] if segment_results['bottom_products'][p]['accuracy'] < 0.6]),
                'low_performing_stores': len([s for s in segment_results['bottom_stores'] if segment_results['bottom_stores'][s]['accuracy'] < 0.6]),
                'improvement_potential': 'Focus on products and stores with accuracy < 60%'
            },
            
            'implementation_roadmap': [
                "Phase 1: Deploy model in pilot stores (highest performing segments)",
                "Phase 2: Optimize threshold based on operational priorities",
                "Phase 3: Roll out to all stores with performance monitoring",
                "Phase 4: Continuous improvement based on segment analysis"
            ]
        },
        
        'risk_assessment': {
            'operational_risks': [
                "Dependency on data quality and completeness",
                "Performance variation across different product categories",
                "Seasonal patterns may require model updates",
                f"Stock-out rate of {business_metrics['inventory_metrics']['stock_out_rate']:.1%} may impact customer satisfaction",
                f"Overstock rate of {business_metrics['inventory_metrics']['overstock_rate']:.1%} ties up inventory space"
            ],
            'mitigation_strategies': [
                "Implement robust data quality monitoring",
                "Set up automated model performance tracking",
                "Establish fallback procedures for system downtime",
                "Create segment-specific optimization strategies",
                "Monitor service level metrics continuously"
            ]
        },
        
        'next_steps': {
            'immediate_actions': [
                "Validate results with business stakeholders",
                "Prepare pilot deployment plan for top-performing segments",
                "Set up monitoring and alerting systems",
                "Train staff on new inventory management process"
            ],
            'medium_term_goals': [
                "Implement A/B testing framework",
                "Develop real-time model updating capabilities",
                "Integrate with existing inventory management systems",
                "Optimize thresholds by product category"
            ],
            'long_term_vision': [
                "Expand to demand forecasting and supply chain optimization",
                "Implement dynamic inventory management",
                "Develop personalized restock recommendations",
                "Create automated inventory optimization system"
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
    print("🚀 ANÁLISIS DE IMPACTO OPERATIVO Y SIMULACIÓN")
    print("="*60)
    
    # 1. Cargar predictor y configuración
    predictor, config = load_predictor_and_config()
    if predictor is None:
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
    threshold_df.to_csv(f'{output_dir}/threshold_sensitivity_analysis.csv', index=False)
    
    with open(f'{output_dir}/business_metrics_complete.json', 'w') as f:
        json.dump(business_metrics, f, indent=2)
    
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
    print(f"   • {plots_dir}/business_error_analysis.png")
    print(f"   • {plots_dir}/threshold_sensitivity_analysis.png")
    print(f"   • {plots_dir}/segment_performance_analysis.png")
    
    print(f"\n🎯 CONCLUSIONES CLAVE:")
    print(f"   • Precisión del modelo: {business_metrics['classification_metrics']['accuracy']:.1%}")
    print(f"   • Nivel de servicio: {business_metrics['inventory_metrics']['service_level']:.1%}")
    print(f"   • Decisiones correctas: {business_metrics['operational_impact']['total_correct_decisions']:,}")
    print(f"   • Productos analizados: {segment_results['product_stats']['total_products_analyzed']}")
    print(f"   • Tiendas analizadas: {segment_results['store_stats']['total_stores_analyzed']}")
    
    # Criterio de recomendación basado en métricas operativas
    deploy_criteria = (
        business_metrics['classification_metrics']['accuracy'] > 0.75 and 
        business_metrics['inventory_metrics']['service_level'] > 0.80
    )
    
    recommendation = "✅ RECOMENDACIÓN: DESPLEGAR EL MODELO" if deploy_criteria else "⚠️ RECOMENDACIÓN: OPTIMIZAR ANTES DE DESPLEGAR"
    print(f"\n{recommendation}")
    
    return business_metrics, results_df, threshold_df, segment_results, executive_report

if __name__ == "__main__":
    business_metrics, results_df, threshold_df, segment_results, executive_report = main()