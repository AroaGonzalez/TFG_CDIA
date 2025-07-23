import requests
import json

# Datos de prueba (IDs existentes en el CSV)
test_data = {
    "ID_ALIAS": 246,
    "ID_LOCALIZACION_COMPRA": 5227
}

# Enviar petición
response = requests.post('http://localhost:8000/predict_from_id', json=test_data)

# Verificar respuesta
if response.status_code == 200:
    result = response.json()
    print("Predicción exitosa:")
    print(f"¿Necesita reposición? {'Sí' if result['necesita_reposicion'] else 'No'}")
    if result['necesita_reposicion']:
        print(f"Cantidad a reponer: {result['cantidad_a_reponer']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)