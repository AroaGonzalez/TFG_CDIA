# 06_model_interpretability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import shap
import lime
import lime.lime_tabular
from joblib import load
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
output_dir = 'results/06_interpretability'
plots_dir = f'{output_dir}/plots'
models_dir = 'models'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def load_data_and_models():
    """Cargar datos y mejores modelos del análisis comparativo"""
    print("\n📊 CARGANDO DATOS Y MODELOS")
    print("-" * 50)
    
    # Cargar datos
    df = pd.read_csv('data/processed/02_features/features_engineered.csv')
    
    # Verificar archivos de resultados
    required_files = [
        'results/03_model_comparison/classification_results.csv',
        'results/03_model_comparison/regression_results.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return None, None, None, None, None
    
    # Cargar resultados para identificar mejores modelos
    class_results = pd.read_csv('results/03_model_comparison/classification_results.csv', index_col=0)
    reg_results = pd.read_csv('results/03_model_comparison/regression_results.csv', index_col=0)
    
    # Identificar mejores modelos
    best_class_model_name = class_results['Test_F1'].idxmax()
    
    # Para regresión, usar solo modelos con transformación logarítmica
    log_models = [model for model in reg_results.index if '(Log)' in model]
    if log_models:
        reg_log_results = reg_results.loc[log_models]
        best_reg_model_name = reg_log_results['Test_R2'].idxmax()
    else:
        best_reg_model_name = reg_results['Test_R2'].idxmax()
    
    print(f"✅ Mejor modelo clasificación: {best_class_model_name}")
    print(f"✅ Mejor modelo regresión: {best_reg_model_name}")
    
    # Obtener features
    features = [col for col in df.columns 
               if col not in ['ID_ALIAS', 'ID_LOCALIZACION_COMPRA', 
                             'necesita_reposicion', 'cantidad_a_reponer', 
                             'log_cantidad_a_reponer']]
    
    print(f"✅ Datos cargados: {df.shape[0]} registros, {len(features)} features")
    
    return df, features, best_class_model_name, best_reg_model_name, class_results

def prepare_interpretability_data(df, features):
    """Preparar datos para análisis de interpretabilidad"""
    print("\n🔧 PREPARANDO DATOS PARA INTERPRETABILIDAD")
    print("-" * 50)
    
    # Seleccionar features numéricas
    X = df[features].select_dtypes(include=['number'])
    
    # Manejar NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Targets
    y_class = df['necesita_reposicion']
    y_reg = df['cantidad_a_reponer']
    y_reg_log = df['log_cantidad_a_reponer']
    
    # Split para interpretabilidad (usar menos datos para eficiencia)
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=0.3, random_state=42, stratify=y_class
    )
    
    # Para regresión, filtrar casos positivos
    mask_train = y_reg[X_train.index] > 0
    mask_test = y_reg[X_test.index] > 0
    
    X_reg_train = X_train[mask_train]
    y_reg_log_train = y_reg_log[X_train.index][mask_train]
    
    X_reg_test = X_test[mask_test]
    y_reg_log_test = y_reg_log[X_test.index][mask_test]
    
    # Escalar datos
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_reg_train_scaled = scaler.transform(X_reg_train)
    X_reg_test_scaled = scaler.transform(X_reg_test)
    
    print(f"✅ Datos preparados:")
    print(f"   • Clasificación: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
    print(f"   • Regresión: {X_reg_train_scaled.shape[0]} train, {X_reg_test_scaled.shape[0]} test")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_class_train': y_class_train,
        'y_class_test': y_class_test,
        'X_reg_train': X_reg_train_scaled,
        'X_reg_test': X_reg_test_scaled,
        'y_reg_log_train': y_reg_log_train,
        'y_reg_log_test': y_reg_log_test,
        'feature_names': X.columns.tolist(),
        'X_train_df': X_train,
        'X_test_df': X_test,
        'scaler': scaler
    }

def train_interpretable_models(data):
    """Entrenar modelos interpretables (árboles de decisión) para comparación"""
    print("\n🌳 ENTRENANDO MODELOS INTERPRETABLES")
    print("-" * 50)
    
    # Árbol de decisión para clasificación
    tree_classifier = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    tree_classifier.fit(data['X_train'], data['y_class_train'])
    
    # Árbol de decisión para regresión
    tree_regressor = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    tree_regressor.fit(data['X_reg_train'], data['y_reg_log_train'])
    
    # Evaluar modelos interpretables
    from sklearn.metrics import f1_score, mean_absolute_error, r2_score
    
    tree_class_pred = tree_classifier.predict(data['X_test'])
    tree_class_f1 = f1_score(data['y_class_test'], tree_class_pred)
    
    tree_reg_pred_log = tree_regressor.predict(data['X_reg_test'])
    tree_reg_pred = np.expm1(tree_reg_pred_log)
    tree_reg_true = np.expm1(data['y_reg_log_test'])
    tree_reg_mae = mean_absolute_error(tree_reg_true, tree_reg_pred)
    tree_reg_r2 = r2_score(tree_reg_true, tree_reg_pred)
    
    print(f"✅ Árbol Clasificación - F1: {tree_class_f1:.4f}")
    print(f"✅ Árbol Regresión - MAE: {tree_reg_mae:.2f}, R²: {tree_reg_r2:.4f}")
    
    return tree_classifier, tree_regressor

def visualize_decision_trees(tree_classifier, tree_regressor, feature_names):
    """Visualizar árboles de decisión"""
    print("\n📊 VISUALIZANDO ÁRBOLES DE DECISIÓN")
    print("-" * 50)
    
    # Visualizar árbol de clasificación (solo primeros niveles)
    plt.figure(figsize=(20, 12))
    plot_tree(tree_classifier, 
             feature_names=feature_names,
             class_names=['No Reponer', 'Reponer'],
             filled=True,
             max_depth=3,  # Limitar profundidad para legibilidad
             fontsize=10)
    plt.title('Árbol de Decisión - Clasificación (3 primeros niveles)')
    plt.savefig(f'{plots_dir}/decision_tree_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualizar árbol de regresión (solo primeros niveles)
    plt.figure(figsize=(20, 12))
    plot_tree(tree_regressor,
             feature_names=feature_names,
             filled=True,
             max_depth=3,  # Limitar profundidad para legibilidad
             fontsize=10)
    plt.title('Árbol de Decisión - Regresión (3 primeros niveles)')
    plt.savefig(f'{plots_dir}/decision_tree_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizaciones de árboles guardadas")

def feature_importance_analysis(tree_classifier, tree_regressor, feature_names):
    """Análisis de importancia de características"""
    print("\n📈 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("-" * 50)
    
    # Importancia de características para clasificación
    class_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': tree_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Importancia de características para regresión
    reg_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': tree_regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualizar top 20 características para cada modelo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Clasificación
    top_20_class = class_importance.head(20)
    sns.barplot(x='importance', y='feature', data=top_20_class, ax=ax1)
    ax1.set_title('Top 20 Características - Clasificación (Árbol de Decisión)')
    ax1.set_xlabel('Importancia')
    
    # Regresión
    top_20_reg = reg_importance.head(20)
    sns.barplot(x='importance', y='feature', data=top_20_reg, ax=ax2)
    ax2.set_title('Top 20 Características - Regresión (Árbol de Decisión)')
    ax2.set_xlabel('Importancia')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/feature_importance_trees.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar importancias
    class_importance.to_csv(f'{output_dir}/feature_importance_classification_tree.csv', index=False)
    reg_importance.to_csv(f'{output_dir}/feature_importance_regression_tree.csv', index=False)
    
    print("✅ Análisis de importancia completado")
    
    return class_importance, reg_importance

def permutation_importance_analysis(tree_classifier, tree_regressor, data):
    """Análisis de importancia por permutación"""
    print("\n🔄 ANÁLISIS DE IMPORTANCIA POR PERMUTACIÓN")
    print("-" * 50)
    
    # Importancia por permutación para clasificación
    perm_importance_class = permutation_importance(
        tree_classifier, data['X_test'], data['y_class_test'],
        n_repeats=10, random_state=42, scoring='f1'
    )
    
    # Importancia por permutación para regresión
    perm_importance_reg = permutation_importance(
        tree_regressor, data['X_reg_test'], data['y_reg_log_test'],
        n_repeats=10, random_state=42, scoring='neg_mean_absolute_error'
    )
    
    # Crear DataFrames
    perm_class_df = pd.DataFrame({
        'feature': data['feature_names'],
        'importance_mean': perm_importance_class.importances_mean,
        'importance_std': perm_importance_class.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    perm_reg_df = pd.DataFrame({
        'feature': data['feature_names'],
        'importance_mean': perm_importance_reg.importances_mean,
        'importance_std': perm_importance_reg.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Visualizar top 15 con barras de error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Clasificación
    top_15_class = perm_class_df.head(15)
    ax1.barh(range(len(top_15_class)), top_15_class['importance_mean'],
             xerr=top_15_class['importance_std'])
    ax1.set_yticks(range(len(top_15_class)))
    ax1.set_yticklabels(top_15_class['feature'])
    ax1.set_title('Top 15 Características por Importancia de Permutación - Clasificación')
    ax1.set_xlabel('Importancia (con desviación estándar)')
    ax1.invert_yaxis()
    
    # Regresión
    top_15_reg = perm_reg_df.head(15)
    ax2.barh(range(len(top_15_reg)), top_15_reg['importance_mean'],
             xerr=top_15_reg['importance_std'])
    ax2.set_yticks(range(len(top_15_reg)))
    ax2.set_yticklabels(top_15_reg['feature'])
    ax2.set_title('Top 15 Características por Importancia de Permutación - Regresión')
    ax2.set_xlabel('Importancia (con desviación estándar)')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar resultados
    perm_class_df.to_csv(f'{output_dir}/permutation_importance_classification.csv', index=False)
    perm_reg_df.to_csv(f'{output_dir}/permutation_importance_regression.csv', index=False)
    
    print("✅ Análisis de importancia por permutación completado")
    
    return perm_class_df, perm_reg_df

def lime_explanations(tree_classifier, tree_regressor, data):
    """Generar explicaciones LIME para casos específicos"""
    print("\n🍋 GENERANDO EXPLICACIONES LIME")
    print("-" * 50)
    
    try:
        # Configurar explainer para clasificación
        explainer_class = lime.lime_tabular.LimeTabularExplainer(
            data['X_train'], 
            feature_names=data['feature_names'],
            class_names=['No Reponer', 'Reponer'],
            mode='classification'
        )
        
        # Seleccionar casos interesantes para explicar
        # Caso 1: Un caso positivo (necesita reposición)
        positive_indices = np.where(data['y_class_test'] == 1)[0]
        if len(positive_indices) > 0:
            positive_idx = positive_indices[0]
            
            # Generar explicación
            explanation_pos = explainer_class.explain_instance(
                data['X_test'][positive_idx], 
                tree_classifier.predict_proba,
                num_features=10
            )
            
            # Guardar explicación como imagen
            explanation_pos.save_to_file(f'{plots_dir}/lime_explanation_positive_case.html')
            
        # Caso 2: Un caso negativo (no necesita reposición)
        negative_indices = np.where(data['y_class_test'] == 0)[0]
        if len(negative_indices) > 0:
            negative_idx = negative_indices[0]
            
            # Generar explicación
            explanation_neg = explainer_class.explain_instance(
                data['X_test'][negative_idx], 
                tree_classifier.predict_proba,
                num_features=10
            )
            
            # Guardar explicación como imagen
            explanation_neg.save_to_file(f'{plots_dir}/lime_explanation_negative_case.html')
        
        print("✅ Explicaciones LIME generadas")
        
    except Exception as e:
        print(f"⚠️ Error al generar explicaciones LIME: {str(e)}")
        print("   Continuando sin LIME...")

def generate_business_rules(tree_classifier, feature_names):
    """Extraer reglas de negocio interpretables del árbol de decisión"""
    print("\n📋 EXTRAYENDO REGLAS DE NEGOCIO")
    print("-" * 50)
    
    from sklearn.tree import export_text
    
    # Extraer reglas del árbol de clasificación
    tree_rules = export_text(tree_classifier, 
                            feature_names=feature_names,
                            max_depth=5)  # Limitar profundidad para legibilidad
    
    # Guardar reglas en archivo
    with open(f'{output_dir}/business_rules.txt', 'w') as f:
        f.write("REGLAS DE NEGOCIO EXTRAÍDAS DEL ÁRBOL DE DECISIÓN\n")
        f.write("=" * 60 + "\n\n")
        f.write("Estas reglas determinan cuándo un producto necesita reposición:\n\n")
        f.write(tree_rules)
    
    # Crear versión simplificada para el informe
    simplified_rules = []
    
    # Analizar las características más importantes del árbol
    feature_importance = tree_classifier.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 características
    
    for idx in top_features_idx:
        if feature_importance[idx] > 0.01:  # Solo características importantes
            feature_name = feature_names[idx]
            importance = feature_importance[idx]
            simplified_rules.append({
                'feature': feature_name,
                'importance': float(importance),
                'description': f'La característica "{feature_name}" es crítica para la decisión de reposición'
            })
    
    # Guardar reglas simplificadas
    with open(f'{output_dir}/simplified_business_rules.json', 'w') as f:
        json.dump({
            'rules': simplified_rules,
            'interpretation': 'Características más importantes para determinar necesidad de reposición',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print("✅ Reglas de negocio extraídas y guardadas")
    
    return simplified_rules

def create_interpretability_dashboard():
    """Crear resumen ejecutivo de interpretabilidad"""
    print("\n📊 CREANDO RESUMEN EJECUTIVO DE INTERPRETABILIDAD")
    print("-" * 50)
    
    # Cargar todos los resultados generados
    results = {}
    
    try:
        # Importancia de características
        results['tree_importance_class'] = pd.read_csv(f'{output_dir}/feature_importance_classification_tree.csv').head(10)
        results['tree_importance_reg'] = pd.read_csv(f'{output_dir}/feature_importance_regression_tree.csv').head(10)
        
        # Importancia por permutación
        results['perm_importance_class'] = pd.read_csv(f'{output_dir}/permutation_importance_classification.csv').head(10)
        results['perm_importance_reg'] = pd.read_csv(f'{output_dir}/permutation_importance_regression.csv').head(10)
        
        # Reglas de negocio
        with open(f'{output_dir}/simplified_business_rules.json', 'r') as f:
            results['business_rules'] = json.load(f)
    except Exception as e:
        print(f"⚠️ Error al cargar algunos resultados: {str(e)}")
    
    # Crear visualización consolidada
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top 10 características por importancia en árbol de clasificación
    if 'tree_importance_class' in results:
        top_10_class = results['tree_importance_class']
        sns.barplot(x='importance', y='feature', data=top_10_class, ax=axes[0,0])
        axes[0,0].set_title('Top 10 Características - Árbol Clasificación')
        axes[0,0].set_xlabel('Importancia')
    
    # Top 10 características por importancia en árbol de regresión
    if 'tree_importance_reg' in results:
        top_10_reg = results['tree_importance_reg']
        sns.barplot(x='importance', y='feature', data=top_10_reg, ax=axes[0,1])
        axes[0,1].set_title('Top 10 Características - Árbol Regresión')
        axes[0,1].set_xlabel('Importancia')
    
    # Top 10 características por importancia de permutación - clasificación
    if 'perm_importance_class' in results:
        top_10_perm_class = results['perm_importance_class']
        sns.barplot(x='importance_mean', y='feature', data=top_10_perm_class, ax=axes[1,0])
        axes[1,0].set_title('Top 10 Características - Permutación Clasificación')
        axes[1,0].set_xlabel('Importancia')
    
    # Top 10 características por importancia de permutación - regresión
    if 'perm_importance_reg' in results:
        top_10_perm_reg = results['perm_importance_reg']
        sns.barplot(x='importance_mean', y='feature', data=top_10_perm_reg, ax=axes[1,1])
        axes[1,1].set_title('Top 10 Características - Permutación Regresión')
        axes[1,1].set_xlabel('Importancia')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/interpretability_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Guardar resumen ejecutivo
    executive_summary = {
        'interpretability_analysis': {
            'objective': 'Análisis de interpretabilidad y explicabilidad de modelos de predicción de stock',
            'methods_used': [
                'Árboles de decisión interpretables',
                'Análisis de importancia de características',
                'Importancia por permutación',
                'Explicaciones LIME para casos específicos',
                'Extracción de reglas de negocio'
            ],
            'key_findings': {
                'most_important_features_classification': results.get('tree_importance_class', {}).head(5).to_dict('records') if 'tree_importance_class' in results else [],
                'most_important_features_regression': results.get('tree_importance_reg', {}).head(5).to_dict('records') if 'tree_importance_reg' in results else [],
                'business_rules_count': len(results.get('business_rules', {}).get('rules', [])),
                'interpretation': 'Los modelos son interpretables y las decisiones pueden explicarse a stakeholders de negocio'
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    with open(f'{output_dir}/executive_summary.json', 'w') as f:
        json.dump(executive_summary, f, indent=2)
    
    print("✅ Resumen ejecutivo de interpretabilidad creado")
    
    return executive_summary

def main():
    print("🚀 ANÁLISIS DE INTERPRETABILIDAD Y EXPLICABILIDAD DE MODELOS")
    print("="*70)
    
    # Cargar datos y identificar mejores modelos
    df, features, best_class_name, best_reg_name, class_results = load_data_and_models()
    if df is None:
        return None
    
    # Preparar datos para interpretabilidad
    data = prepare_interpretability_data(df, features)
    
    # Entrenar modelos interpretables (árboles de decisión)
    tree_classifier, tree_regressor = train_interpretable_models(data)
    
    # Visualizar árboles de decisión
    visualize_decision_trees(tree_classifier, tree_regressor, data['feature_names'])
    
    # Análisis de importancia de características
    class_importance, reg_importance = feature_importance_analysis(
        tree_classifier, tree_regressor, data['feature_names']
    )
    
    # Análisis de importancia por permutación
    perm_class_df, perm_reg_df = permutation_importance_analysis(
        tree_classifier, tree_regressor, data
    )
    
    # Generar explicaciones LIME
    lime_explanations(tree_classifier, tree_regressor, data)
    
    # Extraer reglas de negocio
    business_rules = generate_business_rules(tree_classifier, data['feature_names'])
    
    # Crear dashboard de interpretabilidad
    executive_summary = create_interpretability_dashboard()
    
    print("\n✅ ANÁLISIS DE INTERPRETABILIDAD COMPLETADO")
    print(f"📁 Archivos generados:")
    print(f"   • {output_dir}/feature_importance_*.csv")
    print(f"   • {output_dir}/permutation_importance_*.csv")
    print(f"   • {output_dir}/business_rules.txt")
    print(f"   • {output_dir}/simplified_business_rules.json")
    print(f"   • {output_dir}/executive_summary.json")
    print(f"   • {plots_dir}/decision_tree_*.png")
    print(f"   • {plots_dir}/feature_importance_trees.png")
    print(f"   • {plots_dir}/permutation_importance.png")
    print(f"   • {plots_dir}/interpretability_dashboard.png")
    print(f"   • {plots_dir}/lime_explanation_*.html")
    
    print(f"\n📋 VALOR PARA EL TFG:")
    print(f"✅ Modelos interpretables que complementan los complejos")
    print(f"✅ Explicaciones comprensibles para stakeholders de negocio")
    print(f"✅ Reglas de negocio extraíbles y aplicables")
    print(f"✅ Análisis de robustez mediante importancia por permutación")
    print(f"✅ Explicaciones caso por caso con LIME")
    
    return {
        'tree_models': (tree_classifier, tree_regressor),
        'importance_analysis': (class_importance, reg_importance),
        'permutation_analysis': (perm_class_df, perm_reg_df),
        'business_rules': business_rules,
        'executive_summary': executive_summary
    }

if __name__ == "__main__":
    results = main()