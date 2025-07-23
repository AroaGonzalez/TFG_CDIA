from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/predict_from_id', methods=['POST'])
def predict_from_id():
    data = request.json
    id_alias = data.get('ID_ALIAS')
    id_localizacion = data.get('ID_LOCALIZACION_COMPRA')
    
    # Cargar resultados existentes
    df = pd.read_csv('results/08_business_analysis/prediction_results.csv')
    
    # Buscar la predicción correspondiente
    record = df[(df['ID_ALIAS'] == id_alias) & 
                (df['ID_LOCALIZACION_COMPRA'] == id_localizacion)]
    
    if record.empty:
        return jsonify({"error": "No se encontraron predicciones para los IDs proporcionados"})
    
    # Devolver predicción existente
    response = {
        'necesita_reposicion': bool(record['pred_necesita_reposicion'].values[0]),
        'cantidad_a_reponer': int(round(record['pred_cantidad_a_reponer'].values[0]))
    }
    
    return jsonify(response)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "working"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)