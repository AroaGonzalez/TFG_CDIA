# src/utils/feature_utils.py

def remove_leaky_features(X, verbose=True):
    """
    Elimina características que pueden causar data leakage
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame con todas las características
    verbose : bool, default=True
        Si es True, muestra información sobre las características eliminadas
        
    Returns:
    --------
    X_clean : pandas.DataFrame
        DataFrame sin características con leakage
    clean_features : list
        Lista de nombres de las características limpias
    """
    # Lista completa de características que sabemos que causan leakage
    leaky_features = [
        # Variables originales directas
        'STOCK_RECUENTOS', 
        'CAPACIDAD_MAXIMA',
        
        # Variables derivadas directamente
        'ocupacion_ratio',
        'gap_absoluto',
        'gap_porcentual',
        'gap_relativo',
        'stock_normalizado_alias',
        'stock_relativo_alias',
        'stock_vs_promedio_alias',
        'capacidad_normalizada_loc',
        'capacidad_vs_promedio_loc',
        'cap_relativa_loc',
        'stock_reciente',
        'urgencia_reposicion',
        'nivel_stock',
        'nivel_stock_encoded',
        
        # Variables estadísticas basadas en STOCK_RECUENTOS
        'alias_STOCK_RECUENTOS_count', 
        'alias_STOCK_RECUENTOS_mean', 
        'alias_STOCK_RECUENTOS_std', 
        'alias_STOCK_RECUENTOS_min', 
        'alias_STOCK_RECUENTOS_max',
        'alias_stock_medio',
        'alias_stock_std',
        'loc_STOCK_RECUENTOS_sum', 
        'loc_STOCK_RECUENTOS_mean', 
        'loc_STOCK_RECUENTOS_std',
        'loc_stock_promedio',
        
        # Variables estadísticas basadas en CAPACIDAD_MAXIMA
        'alias_CAPACIDAD_MAXIMA_mean', 
        'alias_CAPACIDAD_MAXIMA_std', 
        'alias_CAPACIDAD_MAXIMA_min', 
        'alias_CAPACIDAD_MAXIMA_max',
        'loc_CAPACIDAD_MAXIMA_count', 
        'loc_CAPACIDAD_MAXIMA_sum',
        'loc_CAPACIDAD_MAXIMA_mean', 
        'loc_CAPACIDAD_MAXIMA_std',
        
        # Interacciones y ratios que usan stock o capacidad
        'alias_utilizacion_media',
        'loc_densidad_stock',
        'loc_ocupacion_media',
        'alias_ocupacion_media',
        'alias_ocupacion_std',
        'ratio_volatilidad_stock',
        'stock_x_diversidad_alias',
        'capacidad_x_popularidad',
        'stock_x_volatilidad',
        'dias_x_stock_ratio',
        'es_tienda_grande_stock_bajo',
        'rango_stock_norm',
        'rango_capacidad_norm',
        'desviacion_ocupacion_alias',
        'desviacion_ocupacion_loc'
    ]
    
    # También detectar variables que contienen palabras clave asociadas a leakage
    leaky_keywords = ['stock', 'capacidad', 'ocupacion', 'gap']
    
    # Encontrar columnas adicionales que podrían tener leakage según keywords
    keyword_based_leaky = [
        col for col in X.columns 
        if any(keyword in col.lower() for keyword in leaky_keywords)
        and col not in leaky_features
    ]
    
    # Si se encuentran columnas sospechosas, añadirlas a la lista
    if keyword_based_leaky and verbose:
        print(f"⚠️ Detectadas {len(keyword_based_leaky)} características adicionales potencialmente con leakage:")
        for col in keyword_based_leaky[:5]:
            print(f"   • {col}")
        if len(keyword_based_leaky) > 5:
            print(f"   • ... y {len(keyword_based_leaky)-5} más")
        
        leaky_features.extend(keyword_based_leaky)
    
    # Obtener la lista de características que realmente están en X
    features_to_remove = [f for f in leaky_features if f in X.columns]
    
    if verbose:
        print(f"🗑️ Features críticos con leakage eliminados: {len(features_to_remove)}")
        if len(features_to_remove) > 0:
            shown_features = features_to_remove[:5]
            print(f"   • Características eliminadas: {', '.join(shown_features)}" + 
                  (f"... y {len(features_to_remove)-5} más" if len(features_to_remove) > 5 else ""))
    
    # Eliminar características
    X_clean = X.drop(columns=features_to_remove, errors='ignore')
    clean_features = X_clean.columns.tolist()
    
    if verbose:
        print(f"✅ Features limpios: {len(clean_features)}")
    
    return X_clean, clean_features


def verify_no_leakage(X, verbose=True):
    """
    Verifica que no haya características con leakage en el dataset
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame a verificar
    verbose : bool, default=True
        Si es True, muestra información sobre las características con leakage encontradas
        
    Returns:
    --------
    is_clean : bool
        True si no se detectó leakage, False en caso contrario
    """
    # Lista de características críticas que definitivamente indican leakage
    critical_features = [
        'STOCK_RECUENTOS', 
        'CAPACIDAD_MAXIMA', 
        'ocupacion_ratio', 
        'gap_absoluto',
        'alias_STOCK_RECUENTOS_mean',
        'alias_CAPACIDAD_MAXIMA_mean'
    ]
    
    # Palabras clave que podrían indicar leakage
    suspicious_keywords = ['stock', 'capacidad', 'ocupacion', 'gap']
    
    # Buscar características críticas
    critical_leakage = [feat for feat in critical_features if feat in X.columns]
    
    # Buscar características sospechosas
    suspicious_features = [
        col for col in X.columns 
        if any(keyword in col.lower() for keyword in suspicious_keywords)
        and col not in critical_leakage
    ]
    
    leakage_detected = len(critical_leakage) > 0
    
    if verbose:
        if critical_leakage:
            print(f"⚠️ ALERTA DE DATA LEAKAGE CRÍTICO: Se encontraron {len(critical_leakage)} variables críticas")
            for feat in critical_leakage:
                print(f"   • {feat}")
        
        if suspicious_features:
            print(f"⚠️ POSIBLE DATA LEAKAGE: Se encontraron {len(suspicious_features)} variables sospechosas")
            for feat in suspicious_features[:5]:
                print(f"   • {feat}")
            if len(suspicious_features) > 5:
                print(f"   • ... y {len(suspicious_features)-5} más")
        
        if not leakage_detected and not suspicious_features:
            print("✅ No se detectó data leakage en las características")
    
    return not leakage_detected


def identify_leaky_features_by_correlation(X, target, correlation_threshold=0.9, verbose=True):
    """
    Identifica características que podrían tener leakage basado en alta correlación con el target
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame con las características
    target : pandas.Series
        Variable objetivo
    correlation_threshold : float, default=0.9
        Umbral de correlación para considerar posible leakage
    verbose : bool, default=True
        Si es True, muestra información sobre las características identificadas
        
    Returns:
    --------
    potential_leaky_features : list
        Lista de características con posible leakage por alta correlación
    """
    import pandas as pd
    import numpy as np
    
    # Asegurar que X solo tenga variables numéricas
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Combinar X y target para calcular correlaciones
    data_combined = pd.concat([X_numeric, pd.Series(target, name='target')], axis=1)
    
    # Calcular correlación con el target
    correlations = data_combined.corr()['target'].abs().sort_values(ascending=False)
    
    # Identificar características con correlación muy alta
    high_corr_features = correlations[correlations > correlation_threshold].index.tolist()
    
    # Eliminar el propio target de la lista
    if 'target' in high_corr_features:
        high_corr_features.remove('target')
    
    if verbose and high_corr_features:
        print(f"⚠️ Se identificaron {len(high_corr_features)} características con posible leakage por alta correlación (>{correlation_threshold})")
        for feat in high_corr_features:
            print(f"   • {feat}: {correlations[feat]:.4f}")
    
    return high_corr_features