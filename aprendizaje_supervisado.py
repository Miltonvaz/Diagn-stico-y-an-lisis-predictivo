import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Prepara variables físicas y partición train/test (70/30)
def preparar_datos_regresion(df):
    col_ac_inv1 = 'inv_01_ac_power_inv_149583'
    col_dc_volt_inv1 = 'inv_01_dc_voltage_inv_149580'
    col_dc_curr_inv1 = 'inv_01_dc_current_inv_149579'
    
    X_reg = df[[col_dc_curr_inv1, col_dc_volt_inv1]].copy()
    X_reg.columns = ['dc_current', 'dc_voltage']
    X_reg['dc_power_physics'] = (X_reg['dc_voltage'] * X_reg['dc_current']) / 1000.0
    y_reg = df[col_ac_inv1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Entrena regresores y calcula R2, MAE y RMSE
def entrenar_y_evaluar_regresores(X_train, X_test, y_train, y_test):
    models = {
        'Regresión Lineal (Directa)': LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42)
    }
    
    results_reg = {}
    for name, model in models.items():
        if name == 'Regresión Lineal (Directa)':
            model.fit(X_train[['dc_current', 'dc_voltage']], y_train)
            preds = model.predict(X_test[['dc_current', 'dc_voltage']])
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results_reg[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'preds': preds}
        
    df_results_reg = pd.DataFrame(results_reg).T[['R2', 'MAE', 'RMSE']]
    print("=== MÉTRICAS DE EVALUACIÓN DE MODELOS SUPERVISADOS ===")
    print(df_results_reg)
    return results_reg

# Dibuja curvas de predicción y distribución de residuos
def graficar_evaluacion_regresor(y_test, preds_rf):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds_rf, alpha=0.3, s=2, color='royalblue', label='Random Forest')
    plt.plot([0, 30], [0, 30], 'r--', linewidth=1.5, label='Ideal (Ajuste Perfecto)')
    plt.title('Comparación: Real vs. Predicho (Random Forest)', fontsize=11, fontweight='bold')
    plt.xlabel('Potencia AC Real (kW)')
    plt.ylabel('Potencia AC Predicha (kW)')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    residuos = y_test - preds_rf
    plt.scatter(preds_rf, residuos, alpha=0.3, s=2, color='darkorange')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2)
    plt.title('Distribución de Residuos (Errores de Predicción)', fontsize=11, fontweight='bold')
    plt.xlabel('Potencia AC Predicha (kW)')
    plt.ylabel('Residuo / Error (kW)')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
