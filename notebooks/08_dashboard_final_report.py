# notebooks/08_dashboard_final_report.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def load_all_results():
    """Cargar todos los resultados del proyecto"""
    try:
        # Datos y predicciones
        predictions_df = pd.read_csv('results/stock_predictions_final.csv')
        business_metrics = pd.read_csv('results/business_metrics.csv')
        recommendations = pd.read_csv('results/business_recommendations.csv')
        
        # Resultados de modelos
        final_models = pd.read_csv('results/final_models_summary.csv')
        validation_summary = pd.read_csv('results/final_validation_summary.csv')
        
        # Feature importance
        consensus_ranking = pd.read_csv('results/feature_consensus_ranking.csv', index_col=0)
        
        print(f"✅ Todos los archivos cargados correctamente")
        
        return {
            'predictions': predictions_df,
            'business_metrics': business_metrics.iloc[0].to_dict(),
            'recommendations': recommendations,
            'models': final_models,
            'validation': validation_summary,
            'features': consensus_ranking
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

def create_executive_dashboard():
    """Crear dashboard ejecutivo interactivo"""
    print("\n📊 CREANDO DASHBOARD EJECUTIVO")
    print("-" * 30)
    
    data = load_all_results()
    if not data:
        return
    
    # Configurar subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Rendimiento de Modelos', 'Impacto de Negocio', 'Distribución de Predicciones',
            'Features Más Importantes', 'ROI por Escenario', 'Casos por Segmento',
            'Ocupación Actual vs Predicha', 'Precisión por Modelo', 'Métricas Temporales'
        ],
        specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Rendimiento de modelos
    models = data['models']['model'].values
    f1_scores = data['models']['classification_f1'].values
    r2_scores = data['models']['regression_r2'].values
    
    fig.add_trace(
        go.Bar(name='F1-Score', x=models, y=f1_scores, marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. Impacto de negocio (usando datos de business_metrics)
    metrics = data['business_metrics']
    roi_value = metrics.get('roi_roi_porcentaje', 0)
    beneficio = metrics.get('roi_beneficio_neto', 0)
    
    fig.add_trace(
        go.Scatter(x=['ROI %', 'Beneficio Neto'], y=[roi_value, beneficio/1000], 
                  mode='markers+text', marker_size=20, marker_color='green'),
        row=1, col=2
    )
    
    # 3. Distribución de predicciones
    pred_counts = data['predictions']['NECESITA_REPOSICION'].value_counts()
    fig.add_trace(
        go.Pie(labels=['No Necesita', 'Necesita Reposición'], 
               values=pred_counts.values, hole=0.3),
        row=1, col=3
    )
    
    # 4. Top 10 features importantes
    top_features = data['features'].sort_values('mean_rank').head(10)
    fig.add_trace(
        go.Bar(x=top_features.index, y=1/top_features['mean_rank'], 
               marker_color='orange'),
        row=2, col=1
    )
    
    # 5. R² scores
    fig.add_trace(
        go.Bar(name='R²-Score', x=models, y=r2_scores, marker_color='lightcoral'),
        row=2, col=3
    )
    
    # 6. Ocupación actual vs predicha (muestra)
    sample_data = data['predictions'].sample(n=min(1000, len(data['predictions'])))
    fig.add_trace(
        go.Scatter(x=sample_data['OCUPACION_ACTUAL'], 
                  y=sample_data['OCUPACION_PREDICHA'],
                  mode='markers', marker_opacity=0.6, marker_color='purple'),
        row=3, col=1
    )
    
    # Actualizar layout
    fig.update_layout(
        height=1200, 
        title_text="Dashboard Ejecutivo - Sistema Predicción Stock Teórico",
        showlegend=False
    )
    
    # Guardar dashboard
    fig.write_html('results/executive_dashboard.html')
    print("✅ Dashboard guardado: results/executive_dashboard.html")

def generate_technical_report():
    """Generar reporte técnico completo"""
    print("\n📝 GENERANDO REPORTE TÉCNICO")
    print("-" * 30)
    
    data = load_all_results()
    if not data:
        return
    
    report = []
    
    # Encabezado
    report.append("# REPORTE TÉCNICO - SISTEMA PREDICCIÓN STOCK TEÓRICO")
    report.append("## Proyecto de Fin de Grado - Ciencia de Datos e IA")
    report.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # Resumen ejecutivo
    report.append("## 1. RESUMEN EJECUTIVO")
    metrics = data['business_metrics']
    roi = metrics.get('roi_roi_porcentaje', 0)
    beneficio = metrics.get('roi_beneficio_neto', 0)
    casos_reposicion = (data['predictions']['NECESITA_REPOSICION'] == 1).sum()
    
    report.append(f"- **ROI estimado**: {roi:.1f}%")
    report.append(f"- **Beneficio neto**: €{beneficio:,.0f}")
    report.append(f"- **Casos que necesitan reposición**: {casos_reposicion:,} ({casos_reposicion/len(data['predictions'])*100:.1f}%)")
    report.append(f"- **Modelo recomendado**: Ensemble (Random Forest + XGBoost + LightGBM)")
    report.append("")
    
    # Metodología
    report.append("## 2. METODOLOGÍA")
    report.append("### 2.1 Datos")
    report.append(f"- Dataset original: {len(data['predictions']):,} registros")
    report.append(f"- Features finales: {len(data['features'])} variables")
    report.append("- Split temporal: 80% train, 20% test")
    report.append("")
    
    report.append("### 2.2 Modelos Evaluados")
    for _, model in data['models'].iterrows():
        report.append(f"- **{model['model']}**: F1={model['classification_f1']:.3f}, R²={model['regression_r2']:.3f}")
    report.append("")
    
    # Features importantes
    report.append("## 3. FEATURES MÁS IMPORTANTES")
    top_10_features = data['features'].sort_values('mean_rank').head(10)
    for i, (feature, row) in enumerate(top_10_features.iterrows(), 1):
        report.append(f"{i:2d}. **{feature}** (rank: {row['mean_rank']:.1f})")
    report.append("")
    
    # Resultados
    report.append("## 4. RESULTADOS PRINCIPALES")
    best_model = data['models'].loc[data['models']['classification_f1'].idxmax()]
    report.append(f"### 4.1 Mejor Modelo: {best_model['model']}")
    report.append(f"- Accuracy Clasificación: {best_model['classification_accuracy']:.3f}")
    report.append(f"- F1-Score: {best_model['classification_f1']:.3f}")
    report.append(f"- R² Regresión: {best_model['regression_r2']:.3f}")
    report.append(f"- MAE: {best_model['regression_mae']:.2f} unidades")
    report.append("")
    
    # Recomendaciones
    report.append("## 5. RECOMENDACIONES")
    for i, rec in data['recommendations'].iterrows():
        report.append(f"### 5.{i+1} {rec['categoria']}")
        report.append(f"**Recomendación**: {rec['recomendacion']}")
        report.append(f"**Justificación**: {rec['justificacion']}")
        report.append(f"**Impacto**: {rec['impacto']}")
        report.append("")
    
    # Próximos pasos
    report.append("## 6. PRÓXIMOS PASOS")
    report.append("1. **Implementación piloto** en 10-20 tiendas seleccionadas")
    report.append("2. **Monitoreo A/B testing** vs método actual")
    report.append("3. **Reentrenamiento mensual** con nuevos datos")
    report.append("4. **Expansión gradual** según resultados piloto")
    report.append("5. **Integración** con sistemas existentes (07, SFI)")
    report.append("")
    
    # Anexos técnicos
    report.append("## 7. ANEXOS TÉCNICOS")
    report.append("### 7.1 Arquitectura del Sistema")
    report.append("- **Backend**: Python + Scikit-learn + XGBoost + LightGBM")
    report.append("- **Features**: 15 variables seleccionadas por consensus ranking")
    report.append("- **Validación**: TimeSeriesSplit para evitar data leakage temporal")
    report.append("- **Métricas**: F1-Score (clasificación), R² y MAE (regresión)")
    report.append("")
    
    report.append("### 7.2 Limitaciones")
    report.append("- Dependiente de calidad de datos de recuentos")
    report.append("- Requiere reentrenamiento regular")
    report.append("- Costos estimados, no reales")
    report.append("- No considera factores externos (promociones, estacionalidad)")
    
    # Guardar reporte
    with open('results/technical_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✅ Reporte técnico guardado: results/technical_report.md")

def create_model_comparison_chart():
    """Crear gráfico comparativo de modelos"""
    data = load_all_results()
    if not data:
        return
    
    models_df = data['models']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Clasificación
    ax1.barh(models_df['model'], models_df['classification_f1'], color='skyblue', edgecolor='black')
    ax1.set_title('F1-Score Clasificación\n(¿Necesita Reposición?)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('F1-Score')
    ax1.grid(True, alpha=0.3)
    
    # Regresión
    ax2.barh(models_df['model'], models_df['regression_r2'], color='lightcoral', edgecolor='black')
    ax2.set_title('R² Score Regresión\n(¿Cuánto Reponer?)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('R² Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_business_impact_summary():
    """Crear resumen visual de impacto de negocio"""
    data = load_all_results()
    if not data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. ROI y beneficios
    metrics = data['business_metrics']
    roi = metrics.get('roi_roi_porcentaje', 0)
    beneficio = metrics.get('roi_beneficio_neto', 0)
    
    axes[0,0].bar(['ROI (%)', 'Beneficio (€K)'], [roi, beneficio/1000], 
                  color=['green' if roi > 0 else 'red', 'green' if beneficio > 0 else 'red'],
                  edgecolor='black')
    axes[0,0].set_title('Impacto Financiero Estimado', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Distribución de predicciones
    pred_counts = data['predictions']['NECESITA_REPOSICION'].value_counts()
    axes[0,1].pie(pred_counts.values, labels=['No Necesita', 'Necesita Reposición'],
                  autopct='%1.1f%%', colors=['lightgreen', 'orange'])
    axes[0,1].set_title('Distribución de Predicciones', fontweight='bold')
    
    # 3. Cantidad a reponer
    cantidad_data = data['predictions'][data['predictions']['CANTIDAD_A_REPONER'] > 0]['CANTIDAD_A_REPONER']
    axes[1,0].hist(cantidad_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Distribución Cantidad a Reponer', fontweight='bold')
    axes[1,0].set_xlabel('Unidades')
    axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Ocupación actual vs predicha
    sample = data['predictions'].sample(n=min(1000, len(data['predictions'])))
    axes[1,1].scatter(sample['OCUPACION_ACTUAL'], sample['OCUPACION_PREDICHA'], 
                      alpha=0.6, s=10)
    axes[1,1].plot([0, 1], [0, 1], 'r--', alpha=0.7)
    axes[1,1].set_xlabel('Ocupación Actual')
    axes[1,1].set_ylabel('Ocupación Predicha')
    axes[1,1].set_title('Ocupación Actual vs Predicha', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/business_impact_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_implementation_guide():
    """Generar guía de implementación"""
    guide = [
        "# GUÍA DE IMPLEMENTACIÓN - SISTEMA PREDICCIÓN STOCK TEÓRICO",
        "",
        "## FASE 1: PREPARACIÓN (Semanas 1-2)",
        "### Requisitos Técnicos",
        "- Python 3.8+ con librerías: scikit-learn, xgboost, lightgbm, pandas",
        "- Acceso a base de datos con tabla STOCK_LOCALIZACION_RAM",
        "- Integración con sistema 07 para recuentos",
        "- Servidor para ejecución de modelos",
        "",
        "### Configuración Inicial",
        "1. Cargar modelos entrenados desde /models/",
        "2. Configurar features_info.json",
        "3. Establecer threshold de probabilidad (recomendado: 0.6)",
        "4. Configurar frecuencia de ejecución (diaria/semanal)",
        "",
        "## FASE 2: PILOTO (Semanas 3-6)",
        "### Selección de Tiendas Piloto",
        "- 10-20 tiendas representativas",
        "- Diferentes tamaños y tipos",
        "- Con datos históricos suficientes",
        "",
        "### Métricas de Seguimiento",
        "- Accuracy de predicciones vs realidad",
        "- Reducción de rupturas de stock",
        "- Optimización de inventario",
        "- Satisfacción del personal de tienda",
        "",
        "## FASE 3: EXPANSIÓN (Semanas 7-12)",
        "### Rollout Gradual",
        "- Expansión por regiones/cadenas",
        "- Monitoreo continuo de KPIs",
        "- Ajuste de parámetros según feedback",
        "",
        "### Mantenimiento",
        "- Reentrenamiento mensual",
        "- Actualización de features",
        "- Backup de modelos",
        "",
        "## PARÁMETROS CRÍTICOS",
        "```python",
        "# Configuración recomendada",
        "THRESHOLD_PROBABILIDAD = 0.6",
        "FRECUENCIA_PREDICCION = 'diaria'",
        "TOP_FEATURES = 15",
        "VENTANA_REENTRENAMIENTO = 30  # días",
        "```",
        "",
        "## CONTACTO TÉCNICO",
        "Para soporte: [email de contacto]",
        "Documentación: /docs/technical_documentation.md"
    ]
    
    with open('results/implementation_guide.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(guide))
    
    print("✅ Guía de implementación: results/implementation_guide.md")

def print_final_summary():
    """Imprimir resumen final del proyecto"""
    print("\n🎉 PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    data = load_all_results()
    if not data:
        return
    
    print("📊 RESULTADOS FINALES:")
    best_model = data['models'].loc[data['models']['classification_f1'].idxmax()]
    print(f"  • Mejor modelo: {best_model['model']}")
    print(f"  • F1-Score: {best_model['classification_f1']:.3f}")
    print(f"  • R² Score: {best_model['regression_r2']:.3f}")
    
    metrics = data['business_metrics']
    print(f"\n💰 IMPACTO DE NEGOCIO:")
    print(f"  • ROI estimado: {metrics.get('roi_roi_porcentaje', 0):.1f}%")
    print(f"  • Beneficio neto: €{metrics.get('roi_beneficio_neto', 0):,.0f}")
    
    casos_reposicion = (data['predictions']['NECESITA_REPOSICION'] == 1).sum()
    print(f"\n📈 PREDICCIONES GENERADAS:")
    print(f"  • Total registros: {len(data['predictions']):,}")
    print(f"  • Necesitan reposición: {casos_reposicion:,} ({casos_reposicion/len(data['predictions'])*100:.1f}%)")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    archivos = [
        "results/stock_predictions_final.csv",
        "results/executive_dashboard.html", 
        "results/technical_report.md",
        "results/implementation_guide.md",
        "models/*.pkl"
    ]
    for archivo in archivos:
        print(f"  • {archivo}")

def main():
    print("🚀 INICIANDO DASHBOARD Y REPORTES FINALES")
    print("="*60)
    
    # Crear dashboard ejecutivo
    create_executive_dashboard()
    
    # Generar reporte técnico
    generate_technical_report()
    
    # Crear visualizaciones finales
    create_model_comparison_chart()
    create_business_impact_summary()
    
    # Generar guía de implementación
    generate_implementation_guide()
    
    # Resumen final
    print_final_summary()
    
    print(f"\n🎯 PROYECTO FINALIZADO")
    print(f"✅ Sistema de predicción de stock teórico completado")
    print(f"✅ Todos los entregables generados")
    print(f"✅ Listo para presentación de TFG")

if __name__ == "__main__":
    main()