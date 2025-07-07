# src/data_loader.py
import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_sql_inserts_to_dataframe(sql_file_path: str) -> pd.DataFrame:
    """
    Convertir archivo SQL con INSERTs de STOCK_LOCALIZACION_RAM a DataFrame
    """
    print(f"📁 Leyendo archivo: {sql_file_path}")
    
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {sql_file_path}")
    
    print("🔍 Extrayendo datos de los INSERTs...")
    
    # Buscar patrón de INSERT
    pattern = r'INSERT INTO.*?STOCK_LOCALIZACION_RAM.*?VALUES\s*\((.*?)\);'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        print("⚠️ No se encontraron INSERTs válidos.")
    
    print(f"✅ Encontrados {len(matches)} registros")
    
    # Definir columnas según estructura RAM
    columns = [
        'ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'ID_AJENO',
        'STOCK_MINIMO', 'STOCK_MAXIMO_LEGACY', 'STOCK_RECUENTOS', 
        'STOCK_TEORICO_LEGACY',
        'FECHA_HORA_EJECUCION_STOCK_RECUENTOS',
        'FECHA_HORA_EJECUCION_STOCK_TEORICO',
        'FECHA_ALTA', 'USUARIO_ALTA', 'FECHA_MODIFICACION',
        'USUARIO_MODIFICACION', 'FECHA_BAJA', 'USUARIO_BAJA',
        'FECHA_HORA_ACTUALIZACION_STOCK_TEORICO',
        'CAPACIDAD_MAXIMA',
        'ID_ALIAS_ANALYTICS',
        'STOCK_RECUENTOS_VALIDADO_BULTOS',
        'STOCK_MINIMO_BULTOS', 
        'CAPACIDAD_MAXIMA_VALIDADA_BULTOS', 
        'RN'
    ]
    
    # Procesar registros
    rows = []
    for i, match in enumerate(matches):
        if i % 5000 == 0:
            print(f"📊 Procesando registro {i:,}")
        
        try:
            values = parse_values_from_insert(match)
            while len(values) < len(columns):
                values.append(None)
            values = values[:len(columns)]
            rows.append(values)
        except Exception as e:
            print(f"⚠️ Error procesando registro {i}: {e}")
            continue
    
    print(f"✅ Procesados {len(rows)} registros exitosamente")
    
    # Crear DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Convertir tipos de datos
    print("🔄 Convirtiendo tipos de datos...")
    numeric_columns = [
        'ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 'ID_AJENO',
        'STOCK_RECUENTOS', 'CAPACIDAD_MAXIMA', 'STOCK_MINIMO'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✅ DataFrame creado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df

def parse_values_from_insert(values_string: str) -> List:
    """Parsear string de valores de INSERT respetando comillas y NULLs"""
    values = []
    current_value = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(values_string):
        char = values_string[i]
        
        if char in ["'", '"'] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ',' and not in_quotes:
            value = current_value.strip()
            if value.upper() == 'NULL':
                values.append(None)
            else:
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                values.append(value)
            current_value = ""
            i += 1
            continue
        
        current_value += char
        i += 1
    
    # Último valor
    if current_value.strip():
        value = current_value.strip()
        if value.upper() == 'NULL':
            values.append(None)
        else:
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            values.append(value)
    
    return values

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Guardar DataFrame procesado como CSV"""
    print(f"💾 Guardando datos procesados en: {output_path}")
    
    # Crear directorio si no existe
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print("✅ Datos guardados exitosamente")

if __name__ == "__main__":
    print("🧪 Módulo data_loader cargado correctamente")