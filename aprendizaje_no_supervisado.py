import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.patches import Ellipse, Circle
import matplotlib.transforms as transforms

# Calcula características promedio y realiza imputación del Inversor 05
def preparar_caracteristicas_inversores(df):
    inverter_features = []
    for i in range(1, 25):
        inv_name = f"Inversor {i:02d}"
        inv_prefix = f"inv_{i:02d}_"
        
        col_ac = [c for c in df.columns if c.startswith(inv_prefix) and 'ac_power' in c][0]
        col_dc_curr = [c for c in df.columns if c.startswith(inv_prefix) and 'dc_current' in c][0]
        
        col_dc_volt_list = [c for c in df.columns if c.startswith(inv_prefix) and 'dc_voltage' in c]
        if col_dc_volt_list:
            val_volt = df[col_dc_volt_list[0]].mean()
        else:
            val_volt = np.nan
            
        mean_ac_power = df[col_ac].mean()
        mean_dc_curr = df[col_dc_curr].mean()
        
        if col_dc_volt_list:
            p_dc = (df[col_dc_volt_list[0]] * df[col_dc_curr]) / 1000.0
            eff = np.where(p_dc > 0.1, df[col_ac] / p_dc, np.nan)
            mean_eff = pd.Series(eff).mean() * 100
        else:
            mean_eff = np.nan
            
        inverter_features.append({
            'Inversor': inv_name,
            'Potencia_AC_kW': mean_ac_power,
            'Corriente_DC_A': mean_dc_curr,
            'Voltaje_DC_V': val_volt,
            'Eficiencia_Percent': mean_eff
        })

    df_features = pd.DataFrame(inverter_features)
    
    healthy_idx = ~df_features['Inversor'].isin(['Inversor 04', 'Inversor 06', 'Inversor 07'])
    mean_volt_healthy = df_features.loc[healthy_idx, 'Voltaje_DC_V'].mean()
    mean_eff_healthy = df_features.loc[healthy_idx, 'Eficiencia_Percent'].mean()

    df_features.loc[df_features['Inversor'] == 'Inversor 05', 'Voltaje_DC_V'] = mean_volt_healthy
    df_features.loc[df_features['Inversor'] == 'Inversor 05', 'Eficiencia_Percent'] = mean_eff_healthy
    
    return df_features

# Ejecuta la reducción de dimensiones con PCA en inversores sanos
def ejecutar_pca(df_healthy, features_cols):
    X_healthy = df_healthy[features_cols]
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)
    
    pca = PCA()
    X_healthy_pca = pca.fit_transform(X_healthy_scaled)
    
    df_healthy['PC1'] = X_healthy_pca[:, 0]
    df_healthy['PC2'] = X_healthy_pca[:, 1]
    
    print("=== ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) ===")
    print(f"Varianza explicada por PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"Varianza acumulada (2 Componentes): {np.sum(pca.explained_variance_ratio_[:2])*100:.2f}%")
    
    return df_healthy, pca, X_healthy_pca

# Dibuja la curva de varianza explicada acumulada
def graficar_varianza_pca(pca):
    varianza_explicada_acum = pca.explained_variance_ratio_.cumsum()
    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(varianza_explicada_acum) + 1), varianza_explicada_acum, marker="o", color="#1f77b4", linewidth=2)
    plt.xlabel("Número de componentes", fontsize=11)
    plt.ylabel("Varianza explicada acumulada", fontsize=11)
    plt.title("Curva de Varianza Explicada - PCA", fontsize=12, fontweight='bold')
    plt.axhline(y=0.80, color="r", linestyle="--", label="Umbral 80%")
    plt.xticks(range(1, 5))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'modelos')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '01_pca_varianza.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

# Grafica curvas del método del codo y de silueta
def optimizar_k_kmeans(X_healthy_pca):
    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_healthy_pca[:, :2])
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_healthy_pca[:, :2], labels))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    
    axs[0].plot(k_range, inertias, marker='o', color='#3d5a80', linewidth=2)
    axs[0].set_title('Método del Codo (Inercia vs K)', fontweight='bold')
    axs[0].set_xlabel('Número de clusters (k)')
    axs[0].set_ylabel('Inercia')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    axs[1].plot(k_range, silhouette_scores, marker='s', color='#ee6c4d', linewidth=2)
    axs[1].set_title('Score Promedio de Silueta vs K', fontweight='bold')
    axs[1].set_xlabel('Número de clusters (k)')
    axs[1].set_ylabel('Silueta Promedio')
    axs[1].set_ylim(0, 1)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Optimización de Clusters K-Means", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'modelos')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '02_clustering_codo_silueta.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

# Compara visualmente inicialización k-means++ vs random
def comparar_inicializaciones(X_healthy_pca):
    inits = ['k-means++', 'random']
    results = {}

    for init in inits:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, init=init)
        labels = kmeans.fit_predict(X_healthy_pca[:, :2])
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_healthy_pca[:, :2], labels)
        results[init] = {
            'labels': labels,
            'inertia': inertia,
            'silhouette': sil_score,
            'centers': kmeans.cluster_centers_
        }

    fig, fig_axs = plt.subplots(1, 2, figsize=(13, 5))
    colors = np.array(['#4a7c59', '#3d5a80', '#ee6c4d'])

    for idx_i, init in enumerate(inits):
        ax = fig_axs[idx_i]
        lbls = results[init]['labels']
        centers = results[init]['centers']
        ax.scatter(X_healthy_pca[:, 0], X_healthy_pca[:, 1], c=colors[lbls], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=150, label='Centroides')
        ax.set_title(f"KMeans init='{init}'\nSilhouette = {results[init]['silhouette']:.3f}", fontweight='bold')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    plt.suptitle("K-Means (K=3) con Diferentes Inicializaciones", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'modelos')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '03_clustering_inicializaciones.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    
    return results

# Grafica fronteras de decisión usando clasificador KNN
def validar_fronteras_knn(X_healthy_pca, best_labels, best_centers, df_healthy):
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_healthy_pca[:, :2], best_labels)
    knn_pred = knn.predict(X_healthy_pca[:, :2])

    acc_knn = accuracy_score(best_labels, knn_pred)
    cm_knn = confusion_matrix(best_labels, knn_pred)

    print("=== EVALUACIÓN CLASIFICADOR KNN (K=5) CON CLUSTERS ===")
    print(f"Exactitud del KNN en PCA: {acc_knn*100:.2f}%")
    print("Matriz de Confusión KNN:")
    print(cm_knn)

    fig = plt.figure(figsize=(8, 6))
    colors_knn = np.array(['#4a7c59', '#3d5a80', '#ee6c4d'])
    plt.scatter(X_healthy_pca[:, 0], X_healthy_pca[:, 1], c=colors_knn[best_labels], alpha=0.75, s=120, edgecolors='black', label="Inversores")
    plt.scatter(best_centers[:, 0], best_centers[:, 1], c='black', marker='X', s=200, label="Centroides")

    for idx, row in df_healthy.reset_index(drop=True).iterrows():
        plt.text(row['PC1'] + 0.05, row['PC2'], row['Inversor'].replace('Inversor ', 'Inv '), fontsize=9, fontweight='semibold')
        
    plt.title(f"Fronteras de Clusters en PCA (KNN Accuracy = {acc_knn*100:.1f}%)", fontweight='bold')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'modelos')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '04_clustering_mejor_modelo.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

# Función auxiliar para trazar elipses de confianza estadística
def draw_confidence_ellipse(x, y, ax, n_std=1.5, facecolor='none', **kwargs):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    if cov[0, 0] * cov[1, 1] == 0:
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Dibuja perfiles de inversores con elipsis de confianza
def graficar_perfiles_kmeans(df_healthy):
    fig, ax = plt.subplots(figsize=(11, 7))
    colors_map = {0: '#3498db', 1: '#2ecc71', 2: '#e67e22'}
    labels_map = {0: 'Perfil 0 (Eficiencia Moderada - Desgaste leve)', 1: 'Perfil 1 (Alta Eficiencia - Líderes)', 2: 'Perfil 2 (Baja Eficiencia - Rezagados)'}

    for cluster_id in sorted(df_healthy['Cluster_Healthy'].unique()):
        cluster_data = df_healthy[df_healthy['Cluster_Healthy'] == cluster_id]
        
        ax.scatter(
            cluster_data['Potencia_AC_kW'], 
            cluster_data['Eficiencia_Percent'], 
            color=colors_map[cluster_id], 
            s=140, 
            edgecolors='black', 
            alpha=0.85,
            label=labels_map[cluster_id]
        )
        
        draw_confidence_ellipse(
            cluster_data['Potencia_AC_kW'], 
            cluster_data['Eficiencia_Percent'], 
            ax, 
            n_std=1.5, 
            facecolor=colors_map[cluster_id], 
            alpha=0.12, 
            edgecolor=colors_map[cluster_id], 
            linestyle='--',
            linewidth=1.5
        )
        
        centroid_x = cluster_data['Potencia_AC_kW'].mean()
        centroid_y = cluster_data['Eficiencia_Percent'].mean()
        ax.scatter(
            centroid_x, centroid_y, 
            color='red', marker='*', s=250, 
            edgecolors='black', 
            label='Centroide de Cluster' if cluster_id == 0 else ""
        )
        ax.text(
            centroid_x + 0.03, centroid_y - 0.2, 
            f"({centroid_x:.2f} kW, {centroid_y:.1f}%)", 
            color='#c0392b', fontsize=9, fontweight='bold'
        )

    for idx, row in df_healthy.iterrows():
        ax.text(
            row['Potencia_AC_kW'] + 0.015, 
            row['Eficiencia_Percent'] - 0.05, 
            row['Inversor'].replace('Inversor ', 'Inv '), 
            fontsize=8.5, 
            fontweight='semibold',
            alpha=0.85
        )

    plt.title('Perfilamiento Operativo de Inversores Saludables (K-Means)\nVisualización de Clusters con Elipses de Confianza (1.5 Dev. Est.)', fontsize=13, fontweight='bold')
    plt.xlabel('Potencia AC Promedio (kW)', fontsize=11)
    plt.ylabel('Eficiencia de Conversión Promedio (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.tight_layout()
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'resultados')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '02_clustering_perfiles.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

# Detecta anomalías espaciales mediante algoritmo DBSCAN
def ejecutar_dbscan_anomalias(df_features, X_cluster_scaled):
    dbscan = DBSCAN(eps=1.5, min_samples=3, metric='euclidean')
    df_features['DBSCAN_labels'] = dbscan.fit_predict(X_cluster_scaled)

    pca_global = PCA(n_components=2)
    X_global_pca = pca_global.fit_transform(X_cluster_scaled)
    df_features['PC1'] = X_global_pca[:, 0]
    df_features['PC2'] = X_global_pca[:, 1]

    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    fig = plt.figure(figsize=(11, 7))
    ax = plt.subplot(1, 1, 1)

    for idx, row in df_features.iterrows():
        label = row['DBSCAN_labels']
        is_core = core_samples_mask[idx]
        
        if label == -1:
            plt.scatter(
                row['PC1'], row['PC2'], 
                color='#e74c3c', marker='X', s=160, edgecolors='black', linewidth=1.2,
                label='Anomalía / Outlier (-1)' if idx == 0 else ""
            )
            plt.text(
                row['PC1'] + 0.12, row['PC2'] - 0.1, 
                row['Inversor'].replace('Inversor ', 'Inv '),
                fontsize=9.5, color='#c0392b', fontweight='bold'
            )
        else:
            color = '#2ecc71' if is_core else '#3498db'
            marker = 'o' if is_core else 's'
            label_text = 'Punto Núcleo (Core)' if is_core else 'Punto Borde'
            
            plt.scatter(
                row['PC1'], row['PC2'], 
                facecolor=color, marker=marker, s=120, edgecolors='black', alpha=0.85,
                label=label_text if (is_core and idx == 1) or (not is_core and idx == 4) else ""
            )
            
            if is_core:
                circle = Circle(
                    (row['PC1'], row['PC2']), radius=1.0, 
                    facecolor='#2ecc71', fill=True, alpha=0.03, 
                    linestyle=':', edgecolor='#27ae60', linewidth=0.8
                )
                ax.add_patch(circle)

    plt.title('Detección Autónoma de Anomalías con DBSCAN\nVisualización en Espacio PCA con Círculos de Densidad (Epsilon)', fontsize=13, fontweight='bold')
    plt.xlabel('Componente Principal 1 (PC1 - Capacidad operativa de generación)', fontsize=11)
    plt.ylabel('Componente Principal 2 (PC2 - Desviaciones de voltaje/eficiencia)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.4)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    output_dir = os.path.join('Reporte-Microservicio-ML-plantilla', 'figures', 'resultados')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '05_clustering_dbscan.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    print("--- DETECCIÓN DE ANOMALÍAS CON DBSCAN (Global - 24 Inversores) ---")
    anomalas = df_features[df_features['DBSCAN_labels'] == -1]['Inversor'].tolist()
    print(f"Inversores clasificados como anomalías (-1 por DBSCAN): {anomalas}")
