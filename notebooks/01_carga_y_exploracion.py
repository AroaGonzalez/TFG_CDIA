# 01_carga_y_exploracion.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from src.data_loader import parse_sql_inserts_to_dataframe

import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("viridis")
pd.set_option('display.max_columns', None)

def prepare_targets(df):
    """Prepara targets de clasificación y regresión de forma consistente"""
    # Target de clasificación
    df['necesita_reposicion'] = (df['STOCK_RECUENTOS'] / df['CAPACIDAD_MAXIMA'] < 0.25).astype(int)
    
    # Target de regresión (solo cantidad positiva)
    nivel_objetivo = 0.80
    stock_objetivo = df['CAPACIDAD_MAXIMA'] * nivel_objetivo
    df['cantidad_a_reponer'] = np.maximum(0, stock_objetivo - df['STOCK_RECUENTOS'])
    
    # Transformación logarítmica (usar esta en todo el código)
    df['log_cantidad_a_reponer'] = np.log1p(df['cantidad_a_reponer'])
    
    return df

def main():
   print("🚀 ANÁLISIS EXPLORATORIO DE DATOS - PREDICCIÓN DE STOCK TEORICO")
   print("="*60)
   
   # PASO 1: Cargar datos desde SQL
   print("\n📊 PASO 1: CARGA DE DATOS")
   print("-" * 30)
   
   sql_file = os.path.join('data', 'raw', 'stock_data.sql')
   
   try:
       df = parse_sql_inserts_to_dataframe(sql_file)
   except Exception as e:
       print(f"❌ Error al cargar datos: {e}")
       return None
   
   print(f"✅ Datos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
   
   # PASO 2: Limpieza inicial de datos
   print("\n🧹 PASO 2: LIMPIEZA DE DATOS")
   print("-" * 30)
   
   # Conversión de fechas
   date_columns = [col for col in df.columns if 'FECHA' in col]
   for col in date_columns:
       if col in df.columns:
           df[col] = pd.to_datetime(df[col], errors='coerce')
   
   # Estadísticas de datos faltantes
   missing_data = df.isnull().sum()
   missing_percent = (missing_data / len(df)) * 100
   missing_stats = pd.DataFrame({
       'Missing Values': missing_data,
       'Percentage': missing_percent
   }).sort_values('Percentage', ascending=False)
   
   print("\nDatos faltantes:")
   print(missing_stats[missing_stats['Missing Values'] > 0].head(10))
   
   # Filtrar registros válidos para análisis
   valid_mask = (
       df['STOCK_RECUENTOS'].notna() &
       df['CAPACIDAD_MAXIMA'].notna() &
       df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].notna() &
       (df['STOCK_RECUENTOS'] >= 0) &
       (df['CAPACIDAD_MAXIMA'] > 0)
   )
   
   df_clean = df[valid_mask].copy()
   
   print(f"\n📊 Registros válidos: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")
   
   # PASO 3: Análisis exploratorio básico
   print("\n🔍 PASO 3: ANÁLISIS EXPLORATORIO BÁSICO")
   print("-" * 30)
   
   # Estadísticas descriptivas de variables principales
   print("\nEstadísticas descriptivas:")
   print(df_clean[['STOCK_RECUENTOS', 'CAPACIDAD_MAXIMA']].describe().round(2))
   
   # Crear directorio para guardar gráficos
   os.makedirs('results/plots', exist_ok=True)
   
   # Análisis de distribución
   plt.figure(figsize=(18, 6))
   
   plt.subplot(1, 3, 1)
   sns.histplot(df_clean['STOCK_RECUENTOS'], kde=True)
   plt.title('Distribución de STOCK_RECUENTOS')
   plt.xlabel('Stock')
   plt.ylabel('Frecuencia')
   
   plt.subplot(1, 3, 2)
   sns.histplot(df_clean['CAPACIDAD_MAXIMA'], kde=True)
   plt.title('Distribución de CAPACIDAD_MAXIMA')
   plt.xlabel('Capacidad')
   plt.ylabel('Frecuencia')
   
   # Ratio de ocupación
   df_clean['ocupacion_ratio'] = df_clean['STOCK_RECUENTOS'] / df_clean['CAPACIDAD_MAXIMA']
   plt.subplot(1, 3, 3)
   sns.histplot(df_clean['ocupacion_ratio'].clip(0, 1), kde=True)
   plt.title('Ratio de Ocupación (Stock/Capacidad)')
   plt.xlabel('Ratio')
   plt.ylabel('Frecuencia')
   
   plt.tight_layout()
   plt.savefig('results/plots/distribucion_variables.png', dpi=300)
   
   # Análisis por alias
   plt.figure(figsize=(12, 6))
   top_aliases = df_clean.groupby('ID_ALIAS')['ID_LOCALIZACION_COMPRA'].count().sort_values(ascending=False).head(10)
   sns.barplot(x=top_aliases.index, y=top_aliases.values)
   plt.title('Top 10 Alias por Frecuencia')
   plt.xlabel('ID Alias')
   plt.ylabel('Número de Localizaciones')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('results/plots/top_aliases.png', dpi=300)
   
   # Análisis temporal
   plt.figure(figsize=(12, 6))
   
   # Agregación por mes
   df_clean['año_mes'] = df_clean['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.to_period('M')
   monthly_stats = df_clean.groupby('año_mes').agg({
       'STOCK_RECUENTOS': 'mean',
       'CAPACIDAD_MAXIMA': 'mean',
       'ocupacion_ratio': 'mean'
   })
   
   monthly_stats.plot(figsize=(12, 6))
   plt.title('Evolución temporal de stock y capacidad')
   plt.ylabel('Valor promedio')
   plt.savefig('results/plots/evolucion_temporal.png', dpi=300)
   
   # PASO 4: Análisis de correlaciones básicas
   print("\n📈 PASO 4: ANÁLISIS DE CORRELACIONES")
   print("-" * 30)
   
   # Crear variables básicas para exploración
   df_clean['gap_absoluto'] = df_clean['CAPACIDAD_MAXIMA'] - df_clean['STOCK_RECUENTOS']
   df_clean['gap_porcentual'] = (df_clean['gap_absoluto'] / df_clean['CAPACIDAD_MAXIMA']).clip(0, 1)
   
   # Crear matriz de correlación
   corr_vars = ['STOCK_RECUENTOS', 'CAPACIDAD_MAXIMA', 'ocupacion_ratio', 'gap_absoluto', 'gap_porcentual']
   corr_matrix = df_clean[corr_vars].corr()
   
   plt.figure(figsize=(10, 8))
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
   plt.title('Matriz de Correlación entre Variables Principales')
   plt.tight_layout()
   plt.savefig('results/plots/correlacion_basica.png', dpi=300)
   
   # PASO 5: Definición de targets para ML
   print("\n🎯 PASO 5: DEFINICIÓN DE TARGETS PARA ML")
   print("-" * 30)
   
   # Aplicar función para crear targets consistentes
   df_clean = prepare_targets(df_clean)
   
   # Información de targets
   print(f"Target clasificación - necesita_reposición:")
   print(f"  • Positivos: {df_clean['necesita_reposicion'].sum()} ({df_clean['necesita_reposicion'].mean():.1%})")
   print(f"  • Negativos: {(1-df_clean['necesita_reposicion']).sum()} ({1-df_clean['necesita_reposicion'].mean():.1%})")
   
   print(f"\nTarget regresión - cantidad_a_reponer:")
   print(f"  • Media: {df_clean['cantidad_a_reponer'].mean():.2f}")
   print(f"  • Mediana: {df_clean['cantidad_a_reponer'].median():.2f}")
   print(f"  • Rango: {df_clean['cantidad_a_reponer'].min():.2f} - {df_clean['cantidad_a_reponer'].max():.2f}")
   
   # Visualizar distribución de targets
   plt.figure(figsize=(16, 6))
   
   plt.subplot(1, 3, 1)
   sns.countplot(x='necesita_reposicion', data=df_clean)
   plt.title('Distribución de Target Clasificación')
   plt.xlabel('Necesita Reposición')
   plt.ylabel('Conteo')
   
   plt.subplot(1, 3, 2)
   sns.histplot(df_clean['cantidad_a_reponer'], kde=True)
   plt.title('Distribución de Target Regresión')
   plt.xlabel('Cantidad a Reponer')
   plt.ylabel('Frecuencia')
   
   plt.subplot(1, 3, 3)
   sns.histplot(df_clean['log_cantidad_a_reponer'], kde=True)
   plt.title('Distribución Log-Transformada')
   plt.xlabel('Log(Cantidad a Reponer + 1)')
   plt.ylabel('Frecuencia')
   
   plt.tight_layout()
   plt.savefig('results/plots/distribucion_targets.png', dpi=300)
   
   # PASO 6: Guardar dataset procesado
   print("\n💾 PASO 6: GUARDAR DATASET PROCESADO")
   print("-" * 30)
   
   # Eliminar variables derivadas que causan leakage
   leakage_vars = ['ocupacion_ratio', 'gap_absoluto', 'gap_porcentual']
   df_final = df_clean.drop(columns=leakage_vars)
   
   # Crear directorio si no existe
   os.makedirs('data/processed', exist_ok=True)
   
   # Guardar dataset completo
   df_final.to_csv('data/processed/stock_data_clean.csv', index=False)
   
   # Generar y guardar metadata
   metadata = {
       'total_records': len(df_final),
       'valid_records': len(df_final),
       'date_min': df_final['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].min().strftime('%Y-%m-%d'),
       'date_max': df_final['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].max().strftime('%Y-%m-%d'),
       'unique_alias': df_final['ID_ALIAS'].nunique(),
       'unique_locations': df_final['ID_LOCALIZACION_COMPRA'].nunique(),
       'reposition_needed_pct': float(df_final['necesita_reposicion'].mean()),
       'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   import json
   with open('data/processed/metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)
   
   print("\n✅ Análisis exploratorio y procesamiento completados")
   print(f"📊 Dataset final: {len(df_final):,} registros")
   print(f"📊 Variables: {df_final.shape[1]}")
   print(f"📊 Archivos guardados:")
   print(f"  • data/processed/stock_data_clean.csv")
   print(f"  • data/processed/metadata.json")
   print(f"  • results/plots/ (múltiples gráficos)")
   
   return df_final

if __name__ == "__main__":
   df_resultado = main()