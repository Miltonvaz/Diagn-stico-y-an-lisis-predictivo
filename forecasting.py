import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Agrupa datos a nivel diario y aplica interpolación lineal
def preparar_datos_forecasting(df):
    healthy_cols = []
    for i in range(1, 25):
        if i in [4, 7]:
            continue
        inv_prefix = f"inv_{i:02d}_"
        col_ac = [c for c in df.columns if c.startswith(inv_prefix) and 'ac_power' in c]
        if col_ac:
            healthy_cols.append(col_ac[0])

    df['planta_total_power_kw'] = df[healthy_cols].sum(axis=1)
    df['planta_energy_mwh'] = df['planta_total_power_kw'] * 5 / 60000

    daily_gen = df.groupby(df['measured_on'].dt.normalize())['planta_energy_mwh'].sum()
    daily_gen = daily_gen[daily_gen > 1.0]

    all_days = pd.date_range(start=daily_gen.index.min(), end=daily_gen.index.max(), freq='D')
    daily_gen_clean = daily_gen.reindex(all_days).interpolate(method='linear')

    train_ts = daily_gen_clean.iloc[:-30]
    test_ts = daily_gen_clean.iloc[-30:]
    return daily_gen_clean, train_ts, test_ts

# Ajusta y grafica pronóstico Holt-Winters de corto plazo (30 días)
def ejecutar_holt_winters_corto_plazo(train_ts, test_ts):
    hw_model = ExponentialSmoothing(train_ts, trend='add', seasonal='add', seasonal_periods=7)
    hw_fit = hw_model.fit()
    forecast_val = hw_fit.forecast(30)

    ts_mae = mean_absolute_error(test_ts, forecast_val)
    ts_rmse = np.sqrt(mean_squared_error(test_ts, forecast_val))
    ts_mape = np.mean(np.abs((test_ts - forecast_val) / test_ts)) * 100

    print("=== EVALUACIÓN PRONÓSTICO HOLT-WINTERS (30 DÍAS) ===")
    print(f"MAE:  {ts_mae:.4f} MWh")
    print(f"RMSE: {ts_rmse:.4f} MWh")
    print(f"MAPE: {ts_mape:.2f}%")

    plt.figure(figsize=(14, 6))
    plt.plot(train_ts.index[-90:], train_ts.values[-90:], label='Histórico de Entrenamiento (Últimos 90 días)', color='black', alpha=0.7)
    plt.plot(test_ts.index, test_ts.values, label='Datos Reales (Test)', color='blue', linewidth=2)
    plt.plot(test_ts.index, forecast_val, label='Pronóstico Holt-Winters', color='red', linestyle='--', linewidth=2)

    plt.title('Pronóstico de Generación de Energía Diaria de la Planta (MWh) - Ventana de Test de 30 Días', fontsize=13, fontweight='bold')
    plt.xlabel('Fecha', fontsize=11)
    plt.ylabel('Generación de Energía (MWh)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return forecast_val

# Pronostica la generación mensual de 2024 destacando extremos
def ejecutar_holt_winters_largo_plazo(daily_gen_clean):
    monthly_gen = daily_gen_clean.resample('ME').sum()
    hw_monthly = ExponentialSmoothing(monthly_gen, trend='add', seasonal='add', seasonal_periods=12)
    hw_monthly_fit = hw_monthly.fit()
    monthly_forecast = hw_monthly_fit.forecast(14)
    forecast_2024 = monthly_forecast.loc['2024-01-01':'2024-12-31']

    print("=== PRONÓSTICO MENSUAL DE GENERACIÓN PARA EL AÑO 2024 ===")
    for fecha, val in forecast_2024.items():
        print(f"{fecha.strftime('%B %Y')}: {val:.2f} MWh")

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_gen.index[-36:], monthly_gen.values[-36:], label='Historial Real Reciente (Últimos 3 años)', color='royalblue', marker='o', linewidth=2)
    plt.plot(forecast_2024.index, forecast_2024.values, label='Pronóstico Holt-Winters 2024 (Largo Plazo)', color='#e67e22', marker='s', linestyle='--', linewidth=2.5)

    mes_pico_val = forecast_2024.max()
    mes_pico_fecha = forecast_2024.idxmax()
    mes_valle_val = forecast_2024.min()
    mes_valle_fecha = forecast_2024.idxmin()

    plt.scatter([mes_pico_fecha, mes_valle_fecha], [mes_pico_val, mes_valle_val], color='red', s=150, zorder=5)
    plt.annotate(f"Pico (Primavera/Verano):\n{mes_pico_val:.1f} MWh", xy=(mes_pico_fecha, mes_pico_val), 
                 xytext=(mes_pico_fecha - pd.Timedelta(days=80), mes_pico_val + 5),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontweight='bold')
    plt.annotate(f"Valle (Otoño/Invierno):\n{mes_valle_val:.1f} MWh", xy=(mes_valle_fecha, mes_valle_val), 
                 xytext=(mes_valle_fecha - pd.Timedelta(days=80), mes_valle_val - 12),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontweight='bold')

    plt.title('Pronóstico de Largo Plazo (Mensual) para el Año 2024:\nIdentificación de Extremos Estacionales (Meses Pico vs. Meses Valle)', fontsize=13, fontweight='bold')
    plt.xlabel('Fecha (Mes)', fontsize=11)
    plt.ylabel('Generación Total Mensual (MWh)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

# Simula escenarios actual vs potencial y grafica pérdida en MWh
def analizar_escenarios_optimizacion(df, daily_gen_clean):
    standard_invs = [2, 3, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    standard_cols = [c for c in df.columns for i in standard_invs if c.startswith(f"inv_{i:02d}_") and 'ac_power' in c]
    standard_cols = list(set(standard_cols))

    df['std_inv_power_kw'] = df[standard_cols].mean(axis=1)
    df['std_inv_energy_mwh'] = df['std_inv_power_kw'] * 5 / 60000
    avg_daily_energy_per_std_inv = df.groupby(df['measured_on'].dt.normalize())['std_inv_energy_mwh'].sum().mean()

    healthy_cols = []
    for i in range(1, 25):
        if i in [4, 7]:
            continue
        inv_prefix = f"inv_{i:02d}_"
        col_ac = [c for c in df.columns if c.startswith(inv_prefix) and 'ac_power' in c]
        if col_ac:
            healthy_cols.append(col_ac[0])

    df['current_plant_power_kw'] = df[healthy_cols].sum(axis=1)
    df['current_plant_energy_mwh'] = df['current_plant_power_kw'] * 5 / 60000
    current_plant_avg = df.groupby(df['measured_on'].dt.normalize())['current_plant_energy_mwh'].sum().mean()

    avg_daily_actual_plant = current_plant_avg
    avg_daily_potential_plant = 24 * avg_daily_energy_per_std_inv

    daily_energy_loss = avg_daily_potential_plant - avg_daily_actual_plant
    monthly_energy_loss = daily_energy_loss * 30

    print("=== ANÁLISIS DE BRECHA DE CAPACIDAD EN LA PLANTA SOLAR ===")
    print(f"Generación Máxima Potencial Diaria:   {avg_daily_potential_plant:.3f} MWh")
    print(f"Generación Real Promedio Diaria:      {avg_daily_actual_plant:.3f} MWh")
    print(f"Pérdida Diaria Neta de Energía:       {daily_energy_loss:.3f} MWh ({ (daily_energy_loss / avg_daily_potential_plant) * 100:.2f}% de la capacidad)")
    print(f"Pérdida Mensual Acumulada (30 días):  {monthly_energy_loss:.3f} MWh")

    plt.figure(figsize=(8, 5))
    escenarios = ['Escenario Actual\n(Con pérdidas - 22 Inv)', 'Escenario Potencial\n(Todos óptimos - 24 Inv)']
    valores = [avg_daily_actual_plant, avg_daily_potential_plant]
    colores = ['#e67e22', '#2ecc71']

    bars = plt.bar(escenarios, valores, color=colores, edgecolor='black', width=0.4)
    plt.ylabel('Generación Diaria (MWh)', fontsize=11)
    plt.title('Escenarios de Optimización de Capacidad Diaria (MWh)', fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.05, f"{height:.3f} MWh", ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, max(valores) * 1.2)
    plt.tight_layout()
    plt.show()
