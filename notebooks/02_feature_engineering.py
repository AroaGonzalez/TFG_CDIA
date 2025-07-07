# 02_feature_engineering.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from src.utils.feature_utils import remove_leaky_features, verify_no_leakage, identify_leaky_features_by_correlation

import warnings
import json
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("viridis")

def load_clean_data():
   """Cargar datos limpios del paso anterior"""
   try:
       df = pd.read_csv('data/processed/stock_data_clean.csv')
       print(f"✅ Datos cargados: {df.shape}")
       return df
   except FileNotFoundError:
       print("❌ Error: Ejecuta primero 01_carga_y_exploracion.py")
       return None

def create_temporal_features(df):
   """Crear características temporales avanzadas"""
   print("\n🕒 CREANDO CARACTERÍSTICAS TEMPORALES")
   print("-" * 30)
   
   # Convertir columnas de fecha a datetime si no lo son
   date_cols = [col for col in df.columns if 'FECHA' in col]
   for col in date_cols:
       if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
           df[col] = pd.to_datetime(df[col], errors='coerce')
   
   # Fecha de referencia (fecha actual o máxima del dataset)
   fecha_ref = df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].max()
   if fecha_ref is pd.NaT:
       fecha_ref = datetime.now()
   
   # Características temporales básicas
   df['dias_desde_recuento'] = (fecha_ref - df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS']).dt.days
   df['semana_del_año'] = df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.isocalendar().week
   df['dia_del_año'] = df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.dayofyear
   df['es_fin_semana'] = (df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.dayofweek >= 5).astype(int)
   
   # Características cíclicas (seno y coseno para representar periodicidad)
   df['mes_sen'] = np.sin(2 * np.pi * df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.month / 12)
   df['mes_cos'] = np.cos(2 * np.pi * df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.month / 12)
   df['dia_semana_sen'] = np.sin(2 * np.pi * df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.dayofweek / 7)
   df['dia_semana_cos'] = np.cos(2 * np.pi * df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.dayofweek / 7)
   
   # Indicadores de temporada
   df['es_temporada_alta'] = ((df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.month >= 11) | 
                            (df['FECHA_HORA_EJECUCION_STOCK_RECUENTOS'].dt.month <= 1)).astype(int)
   
   # Indicadores de actualidad de datos
   df['dato_muy_reciente'] = (df['dias_desde_recuento'] <= 3).astype(int)
   df['dato_reciente'] = (df['dias_desde_recuento'] <= 7).astype(int)
   df['dato_medio'] = ((df['dias_desde_recuento'] > 7) & (df['dias_desde_recuento'] <= 30)).astype(int)
   df['dato_antiguo'] = (df['dias_desde_recuento'] > 30).astype(int)
   
   # Características de tiempo normalizado
   max_dias = df['dias_desde_recuento'].max()
   if max_dias > 0:
       df['antiguedad_normalizada'] = df['dias_desde_recuento'] / max_dias
   else:
       df['antiguedad_normalizada'] = 0
   
   print(f"✅ Características temporales creadas: {len(df.columns) - len(date_cols)}")
   return df

def create_alias_features(df):
   """Crear características por alias sin leakage"""
   print("\n🏷️ CREANDO CARACTERÍSTICAS POR ALIAS (SIN LEAKAGE)")
   print("-" * 30)
   
   # Eliminar características de alias existentes para evitar duplicados
   alias_cols = [col for col in df.columns if col.startswith('alias_')]
   if alias_cols:
       df = df.drop(columns=alias_cols)
   
   # Agrupar por alias y calcular estadísticas (sin usar STOCK_RECUENTOS)
   alias_stats = df.groupby('ID_ALIAS').agg({
       'ID_LOCALIZACION_COMPRA': 'count',
       'CAPACIDAD_MAXIMA': ['mean', 'std', 'min', 'max', 'median'],
       'dias_desde_recuento': ['mean', 'min', 'max'],
       'es_fin_semana': 'mean',
       'es_temporada_alta': 'mean'
   }).round(3)
   
   # Aplanar nombres de columnas
   alias_stats.columns = ['_'.join(col).strip() for col in alias_stats.columns.values]
   alias_stats = alias_stats.add_prefix('alias_')
   
   # Renombrar columnas para claridad
   rename_dict = {
       'alias_ID_LOCALIZACION_COMPRA_count': 'alias_frecuencia',
       'alias_es_fin_semana_mean': 'alias_pct_fin_semana',
       'alias_es_temporada_alta_mean': 'alias_pct_temporada_alta'
   }
   alias_stats = alias_stats.rename(columns=rename_dict)
   
   # Características derivadas de alias
   alias_stats['alias_coef_variacion'] = (
       alias_stats['alias_CAPACIDAD_MAXIMA_std'] / 
       alias_stats['alias_CAPACIDAD_MAXIMA_mean']
   ).replace([np.inf, -np.inf], np.nan).fillna(0)
   
   # Categorización de alias por popularidad
   alias_stats['alias_popularidad'] = pd.qcut(
       alias_stats['alias_frecuencia'], 
       q=3, 
       labels=['Baja', 'Media', 'Alta']
   ).astype(str)
   
   # Merge con el DataFrame original
   df = df.merge(alias_stats, left_on='ID_ALIAS', right_index=True, how='left')
   
   print(f"✅ Características por alias creadas: {len(alias_stats.columns)}")
   return df

def create_location_features(df):
   """Crear características por localización sin leakage"""
   print("\n🏪 CREANDO CARACTERÍSTICAS POR LOCALIZACIÓN (SIN LEAKAGE)")
   print("-" * 30)
   
   # Eliminar características de localización existentes
   loc_cols = [col for col in df.columns if col.startswith('loc_')]
   if loc_cols:
       df = df.drop(columns=loc_cols)
   
   # Agrupar por localización y calcular estadísticas (sin usar STOCK_RECUENTOS)
   loc_stats = df.groupby('ID_LOCALIZACION_COMPRA').agg({
       'ID_ALIAS': 'nunique',
       'CAPACIDAD_MAXIMA': ['count', 'sum', 'mean', 'std'],
       'dias_desde_recuento': ['mean', 'min', 'max']
   }).round(3)
   
   # Aplanar nombres de columnas
   loc_stats.columns = ['_'.join(col).strip() for col in loc_stats.columns.values]
   loc_stats = loc_stats.add_prefix('loc_')
   
   # Renombrar columnas para claridad
   rename_dict = {
       'loc_ID_ALIAS_nunique': 'loc_diversidad_alias',
       'loc_CAPACIDAD_MAXIMA_count': 'loc_num_registros',
       'loc_CAPACIDAD_MAXIMA_sum': 'loc_capacidad_total'
   }
   loc_stats = loc_stats.rename(columns=rename_dict)
   
   # Características derivadas
   loc_stats['loc_capacidad_por_alias'] = (
       loc_stats['loc_capacidad_total'] / 
       loc_stats['loc_diversidad_alias']
   ).replace([np.inf, -np.inf], np.nan).fillna(0)
   
   # Categorización por tamaño con manejo de duplicados
   try:
       loc_stats['loc_tamaño'] = pd.qcut(
           loc_stats['loc_capacidad_total'], 
           q=3, 
           labels=['Pequeña', 'Mediana', 'Grande'],
           duplicates='drop'
       ).astype(str)
   except Exception as e:
       print(f"   • Advertencia en categorización de tamaño: {e}")
       # Alternativa: usar percentiles manuales
       percentiles = loc_stats['loc_capacidad_total'].quantile([0.33, 0.67]).values
       loc_stats['loc_tamaño'] = pd.cut(
           loc_stats['loc_capacidad_total'],
           bins=[0, percentiles[0], percentiles[1], float('inf')],
           labels=['Pequeña', 'Mediana', 'Grande'],
           include_lowest=True
       ).astype(str)
   
   # Categorización por diversidad con manejo de duplicados
   try:
       loc_stats['loc_diversidad'] = pd.qcut(
           loc_stats['loc_diversidad_alias'], 
           q=3, 
           labels=['Baja', 'Media', 'Alta'],
           duplicates='drop'
       ).astype(str)
   except Exception as e:
       print(f"   • Advertencia en categorización de diversidad: {e}")
       # Alternativa: usar percentiles manuales
       percentiles = loc_stats['loc_diversidad_alias'].quantile([0.33, 0.67]).values
       loc_stats['loc_diversidad'] = pd.cut(
           loc_stats['loc_diversidad_alias'],
           bins=[0, percentiles[0], percentiles[1], float('inf')],
           labels=['Baja', 'Media', 'Alta'],
           include_lowest=True
       ).astype(str)
   
   # Merge con el DataFrame original
   df = df.merge(loc_stats, left_on='ID_LOCALIZACION_COMPRA', right_index=True, how='left')
   
   print(f"✅ Características por localización creadas: {len(loc_stats.columns)}")
   return df

def create_interaction_features(df):
   """Crear características de interacción sin leakage"""
   print("\n🔗 CREANDO CARACTERÍSTICAS DE INTERACCIÓN (SIN LEAKAGE)")
   print("-" * 30)
   
   # Eliminar interacciones previas que podrían tener leakage
   interaction_cols = [col for col in df.columns if '_x_' in col]
   if interaction_cols:
       df = df.drop(columns=interaction_cols)
   
   # 1. Interacciones entre tiempo y capacidad
   df['dias_x_capacidad_max'] = df['dias_desde_recuento'] * df['CAPACIDAD_MAXIMA']
   df['antiguedad_x_capacidad'] = df['antiguedad_normalizada'] * df['CAPACIDAD_MAXIMA']
   
   # 2. Interacciones entre alias y localización
   df['diversidad_x_capacidad'] = df['loc_diversidad_alias'] * df['CAPACIDAD_MAXIMA']
   df['frecuencia_x_capacidad'] = df['alias_frecuencia'] * df['CAPACIDAD_MAXIMA']
   
   # 3. Interacciones categóricas
   if 'alias_popularidad' in df.columns and 'loc_tamaño' in df.columns:
       # Combinar categorías en nuevas features
       df['es_alias_popular_loc_grande'] = (
           (df['alias_popularidad'] == 'Alta') & 
           (df['loc_tamaño'] == 'Grande')
       ).astype(int)
       
       df['es_alias_popular_loc_pequeña'] = (
           (df['alias_popularidad'] == 'Alta') & 
           (df['loc_tamaño'] == 'Pequeña')
       ).astype(int)
   
   # 4. Relaciones compuestas
   if 'loc_capacidad_total' in df.columns and 'alias_CAPACIDAD_MAXIMA_mean' in df.columns:
       df['capacidad_relativa_tienda'] = (
           df['CAPACIDAD_MAXIMA'] / 
           df['loc_capacidad_total']
       ).replace([np.inf, -np.inf], np.nan).fillna(0)
       
       df['capacidad_relativa_alias'] = (
           df['CAPACIDAD_MAXIMA'] / 
           df['alias_CAPACIDAD_MAXIMA_mean']
       ).replace([np.inf, -np.inf], np.nan).fillna(1)
   
   # 5. Interacciones con tiempo
   df['reciente_x_capacidad'] = df['dato_reciente'] * df['CAPACIDAD_MAXIMA']
   df['antiguo_x_capacidad'] = df['dato_antiguo'] * df['CAPACIDAD_MAXIMA']
   
   new_features = [col for col in df.columns if col not in interaction_cols and 
                  ('_x_' in col or 'relativa' in col)]
   
   print(f"✅ Características de interacción creadas: {len(new_features)}")
   return df

def apply_transformations(df):
   """Aplicar transformaciones a variables numéricas"""
   print("\n🔄 APLICANDO TRANSFORMACIONES")
   print("-" * 30)

   stock_related_cols = [col for col in df.columns if 'stock' in col.lower() or 'recuent' in col.lower()]
   if stock_related_cols:
       print(f"⚠️ Detectadas {len(stock_related_cols)} columnas relacionadas con stock que deben eliminarse:")
       for col in stock_related_cols[:5]:
           print(f"   • {col}")
       if len(stock_related_cols) > 5:
           print(f"   • ... y {len(stock_related_cols)-5} más")
       df = df.drop(columns=stock_related_cols)
   
   # 1. Transformaciones logarítmicas para variables con alta skewness
   log_candidates = ['CAPACIDAD_MAXIMA', 'alias_frecuencia', 'loc_capacidad_total']
   
   for col in log_candidates:
       if col in df.columns and df[col].min() >= 0:
           new_col = f'{col}_log'
           df[new_col] = np.log1p(df[col])
           skew_before = df[col].skew()
           skew_after = df[new_col].skew()
           print(f"   • Log({col}): skew antes={skew_before:.2f}, después={skew_after:.2f}")
   
   # 2. Transformaciones Box-Cox para normalizar distribuciones
   power_candidates = ['alias_CAPACIDAD_MAXIMA_std', 'loc_CAPACIDAD_MAXIMA_std']
   
   for col in power_candidates:
       if col in df.columns and (df[col] > 0).all():
           try:
               pt = PowerTransformer(method='box-cox')
               new_col = f'{col}_boxcox'
               df[new_col] = pt.fit_transform(df[[col]])
               skew_before = df[col].skew()
               skew_after = df[new_col].skew()
               print(f"   • BoxCox({col}): skew antes={skew_before:.2f}, después={skew_after:.2f}")
           except Exception:
               pass
   
   # 3. Raíz cuadrada para variables con asimetría moderada
   sqrt_candidates = ['alias_coef_variacion', 'dias_desde_recuento']
   
   for col in sqrt_candidates:
       if col in df.columns and df[col].min() >= 0:
           new_col = f'{col}_sqrt'
           df[new_col] = np.sqrt(df[col])
           skew_before = df[col].skew()
           skew_after = df[new_col].skew()
           print(f"   • Sqrt({col}): skew antes={skew_before:.2f}, después={skew_after:.2f}")
   
   return df

def encode_categorical_features(df):
    """Codificar características categóricas de forma controlada"""
    print("\n🔢 CODIFICANDO VARIABLES CATEGÓRICAS")
    print("-" * 30)
    
    # Detectar variables categóricas
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' 
                       or df[col].dtype.name == 'category']
    
    # Filtrar variables ID y columnas ya codificadas
    exclude_patterns = ['ID_', '_Alta', '_Baja', '_Grande', '_Pequeña', 'RN', 'USUARIO']
    categorical_cols = [col for col in categorical_cols 
                       if not any(pattern in col for pattern in exclude_patterns)]
    
    if not categorical_cols:
        print("   • No se encontraron variables categóricas para codificar")
        return df
    
    # One-hot encoding controlado
    for col in categorical_cols:
        # Verificar si ya existen dummies para esta columna
        existing_dummies = [c for c in df.columns if c.startswith(f"{col}_")]
        if existing_dummies:
            print(f"   • {col}: dummies ya existentes ({len(existing_dummies)})")
            continue
        
        # Verificar cardinalidad para evitar demasiadas dummies
        n_values = df[col].nunique()
        if n_values > 10:
            print(f"   • {col}: demasiados valores únicos ({n_values}), se omite")
            continue
        
        # Crear nuevas dummies
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        
        print(f"   • {col}: {dummies.shape[1]} dummies creadas")
    
    return df

def handle_outliers(df):
   """Manejar outliers en variables numéricas"""
   print("\n📏 MANEJANDO OUTLIERS")
   print("-" * 30)
   
   # Seleccionar variables numéricas, excluyendo IDs y variables target
   exclude_patterns = ['ID_', 'necesita_reposicion', 'cantidad_a_reponer']
   numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
   numeric_cols = [col for col in numeric_cols 
                  if not any(pattern in col for pattern in exclude_patterns)]
   
   # Aplicar winsorización robusta
   for col in numeric_cols:
       if df[col].isna().sum() > 0 or len(df[col].unique()) <= 1:
           continue
       
       # Calcular percentiles robustos
       q1 = df[col].quantile(0.01)
       q3 = df[col].quantile(0.99)
       iqr = q3 - q1
       
       lower_bound = q1 - 1.5 * iqr
       upper_bound = q3 + 1.5 * iqr
       
       # Contar outliers
       outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
       
       if outliers > 0 and outliers / len(df) < 0.1:  # Solo si hay menos del 10% de outliers
           # Crear versión winsorizada
           df[f'{col}_win'] = df[col].clip(lower=lower_bound, upper=upper_bound)
           print(f"   • {col}: {outliers} outliers ({outliers/len(df):.1%}) winsorizado")
   
   return df

def select_final_features(df):
   """Seleccionar características finales"""
   print("\n🎯 SELECCIONANDO CARACTERÍSTICAS FINALES")
   print("-" * 30)
   
   # Verificar targets disponibles
   targets = ['necesita_reposicion', 'cantidad_a_reponer', 'log_cantidad_a_reponer']
   available_targets = [t for t in targets if t in df.columns]
   
   if not available_targets:
       print("⚠️ No se encontraron targets. Verifica el script de exploración.")
       return df, []
   
   # IDs para referencia
   id_cols = ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA']
   
   # Verificación final de leakage
   try:
       # Eliminar características con leakage
       X_clean = df.drop(columns=available_targets + id_cols)
       X_clean, clean_features = remove_leaky_features(X_clean)
       
       # Verificación adicional por correlación con target
       if 'cantidad_a_reponer' in available_targets:
           leaky_by_corr = identify_leaky_features_by_correlation(
               X_clean, 
               df['cantidad_a_reponer'], 
               correlation_threshold=0.85
           )
           
           if leaky_by_corr:
               X_clean = X_clean.drop(columns=leaky_by_corr)
               clean_features = [f for f in clean_features if f not in leaky_by_corr]
       
       # Verificar que no queden variables con leakage
       is_clean = verify_no_leakage(X_clean)
       if not is_clean:
           print("⚠️ Aún pueden existir variables con leakage. Se recomienda revisión manual.")
       
       # Crear dataset final
       features_final = [col for col in clean_features if col not in available_targets and col not in id_cols]
       
       # Dataset final
       final_cols = id_cols + features_final + available_targets
       df_final = pd.concat([df[id_cols], X_clean, df[available_targets]], axis=1)
       
   except Exception as e:
       print(f"⚠️ Error en la verificación de leakage: {e}")
       print("   Realizando limpieza manual básica...")
       
       # Limpieza manual básica
       leaky_patterns = ['STOCK_RECUENTOS', 'ocupacion_ratio', 'gap_']
       leaky_cols = [col for col in df.columns 
                    if any(pattern in col.lower() for pattern in leaky_patterns)]
       
       df_clean = df.drop(columns=leaky_cols)
       features_final = [col for col in df_clean.columns 
                        if col not in available_targets and col not in id_cols]
       
       final_cols = id_cols + features_final + available_targets
       df_final = df_clean[final_cols].copy()
   
   # Manejar valores faltantes
   for col in df_final.columns:
       if df_final[col].isna().sum() > 0:
           if pd.api.types.is_numeric_dtype(df_final[col]):
               df_final[col] = df_final[col].fillna(df_final[col].median())
           else:
               df_final[col] = df_final[col].fillna(df_final[col].mode().iloc[0])
   
   print(f"✅ Características finales seleccionadas: {len(features_final)}")
   print(f"✅ Dataset final: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
   
   return df_final, features_final

def analyze_feature_importance(df, features, targets):
   """Analizar importancia de características mediante correlación"""
   print("\n📊 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
   print("-" * 30)
   
   # Crear carpeta para gráficos
   os.makedirs('results/plots', exist_ok=True)
   
   # Características numéricas
   numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
   
   # Analizar por target
   for target in targets:
       if target not in df.columns:
           continue
       
       print(f"\n🎯 Análisis para target: {target}")
       
       # Calcular correlaciones
       corr = df[numeric_features + [target]].corr()[target].sort_values(ascending=False)
       
       # Mostrar top 10 correlaciones
       top_corr = corr.drop(target).head(10)
       print("Top 10 características por correlación:")
       for feat, val in top_corr.items():
           print(f"   • {feat}: {val:.3f}")
       
       # Visualizar correlaciones
       plt.figure(figsize=(12, 8))
       top_20 = corr.drop(target).abs().sort_values(ascending=False).head(20)
       sns.barplot(x=top_20.values, y=top_20.index)
       plt.title(f'Top 20 Características por Correlación con {target}')
       plt.xlabel('Correlación Absoluta')
       plt.tight_layout()
       plt.savefig(f'results/plots/correlacion_{target}.png', dpi=300)
   
   # Matriz de correlación entre features
   plt.figure(figsize=(16, 14))
   top_vars = numeric_features[:20] if len(numeric_features) > 20 else numeric_features
   corr_matrix = df[top_vars].corr()
   
   mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
   sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
               linewidths=.5, center=0, square=True)
   plt.title('Matriz de Correlación entre Características')
   plt.tight_layout()
   plt.savefig('results/plots/matriz_correlacion.png', dpi=300)
   
   print(f"\n✅ Análisis de importancia guardado en results/plots/")
   
   return top_corr

def main():
   print("🚀 INICIANDO FEATURE ENGINEERING AVANZADO")
   print("="*60)
   
   # Cargar datos limpios
   df = load_clean_data()
   if df is None:
       return None
   
   # Aplicar ingeniería de características
   df = create_temporal_features(df)
   df = create_alias_features(df)
   df = create_location_features(df)
   df = create_interaction_features(df)
   df = apply_transformations(df)
   df = encode_categorical_features(df)
   df = handle_outliers(df)
   
   # Seleccionar características finales (eliminando leakage)
   df_final, features_final = select_final_features(df)
   
   # Analizar importancia de características
   targets = ['necesita_reposicion', 'cantidad_a_reponer', 'log_cantidad_a_reponer']
   available_targets = [t for t in targets if t in df_final.columns]
   
   if available_targets:
       top_features = analyze_feature_importance(df_final, features_final, available_targets)
   
   # Guardar dataset final
   os.makedirs('data/processed', exist_ok=True)
   df_final.to_csv('data/processed/features_engineered.csv', index=False)
   
   # Guardar metadata
   metadata = {
       'features': features_final,
       'total_features': len(features_final),
       'numeric_features': len([f for f in features_final if pd.api.types.is_numeric_dtype(df_final[f])]),
       'categorical_features': len([f for f in features_final if not pd.api.types.is_numeric_dtype(df_final[f])]),
       'targets': available_targets,
       'dataset_shape': df_final.shape,
       'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   }
   
   with open('data/processed/feature_metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)
   
   print(f"\n✅ Feature engineering completado")
   print(f"   • Dataset final: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
   print(f"   • Archivos guardados:")
   print(f"     - data/processed/features_engineered.csv")
   print(f"     - data/processed/feature_metadata.json")
   
   return df_final, features_final

if __name__ == "__main__":
   df_final, features = main()