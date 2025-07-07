# 🎯 TFG: Predicción de Stock con Machine Learning

## 📊 Resultados Principales

### 🏆 Mejores Algoritmos
- **Clasificación:** LightGBM - 81.8% accuracy
- **Regresión:** Random Forest - R² = 0.130

### 📈 Dataset
- **Total registros:** 4,530
- **Necesitan reposición:** 3,270 (72.2%)
- **Alias únicos:** 121
- **Tiendas únicas:** 2361

## 🔍 Clasificación (¿Necesita reposición?)

| Algoritmo | CV Score | Test Accuracy |
|-----------|----------|---------------|
| LightGBM | 0.820 | **0.818** |
| XGBoost | 0.810 | **0.814** |
| Random Forest | 0.790 | **0.798** |
| KNN | 0.756 | **0.767** |
| SVM | 0.752 | **0.744** |
| Logistic Regression | 0.734 | **0.726** |

## 📊 Regresión (¿Cuánto stock?)

| Algoritmo | MAE | RMSE | R² |
|-----------|-----|------|-----|
| Random Forest | 1783.7 | 13625.3 | **0.130** |
| XGBoost | 1897.2 | 13644.7 | **0.127** |
| LightGBM | 2193.5 | 13856.5 | **0.100** |
| Linear Regression | 1873.6 | 13901.1 | **0.094** |
| Ridge | 1873.6 | 13901.2 | **0.094** |
| SVR | 1979.7 | 14694.2 | **-0.012** |

## 🎯 Conclusiones Clave

- ✅ **Gradient Boosting domina:** XGBoost y LightGBM superan métodos tradicionales
- ✅ **Clasificación exitosa:** 81.8% accuracy vs R² = 0.130 en regresión  
- ✅ **Dataset balanceado:** 72.2% casos positivos
- ⚠️ **Regresión compleja:** Predicción exacta de cantidad es desafiante
- 🔧 **Features clave:** stock_ratio, STOCK_RECUENTOS, gap_to_max

## 🚀 Implementación Recomendada

1. **Usar LightGBM** para clasificación en producción
2. **Modelo híbrido:** Clasificación + regresión condicional
3. **Monitoreo continuo** de performance en producción
4. **Incorporar features temporales** para mejorar predicción

---
*Reporte generado automáticamente - 22/06/2025*
