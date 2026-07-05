# Diagnóstico y Análisis Predictivo de una Planta Solar Fotovoltaica

**Alumno:** Milton De Jesús Vázquez Pérez
**Materia:** Minería de Datos — Universidad Politécnica de Chiapas
**Evaluación:** Diagnóstico y análisis predictivo

---

## ¿Qué contiene este repositorio?

| Archivo / Carpeta                       | Descripción                                        |
| --------------------------------------- | --------------------------------------------------- |
| `notebook.ipynb`                      | Notebook principal con todo el análisis y modelado |
| `aprendizaje_no_supervisado.py`       | Módulo de PCA, K-Means y DBSCAN                    |
| `aprendizaje_supervisado.py`          | Módulo de regresión supervisada (Random Forest)   |
| `forecasting.py`                      | Módulo de pronóstico estacional (Holt-Winters)    |
| `requirements.txt`                    | Librerías de Python necesarias                     |
| `Reporte-Microservicio-ML-plantilla/` | Reporte escrito en LaTeX                            |

---

## ⚠️ El archivo de datos no está incluido en el repositorio

Paso 1 — Colocar el archivo de datos

Coloca el archivo dentro de la carpeta `data/` del proyecto con el siguiente nombre:

```
Diagnóstico y análisis predictivo/
│
├── data/
│   └── 2107_electrical_data.csv   ← aquí va el archivo
│
├── notebook.ipynb
└── ...
```

### Paso 2 — Instalar las dependencias

```bash
pip install -r requirements.txt
```

### Paso 3 — Abrir y correr el notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Estructura del análisis

```
1. Limpieza y preparación de datos
2. Análisis Exploratorio de Datos (EDA)
   ├── Boxplots y distribuciones por inversor
   ├── Curvas de conversión AC/DC
   ├── Curvas MPPT (DC Current vs DC Voltage)
   └── Matrices de correlación
3. Reducción de dimensiones (PCA)
4. Clustering no supervisado (K-Means + DBSCAN)
5. Regresión supervisada (Random Forest)
6. Pronóstico estacional (Holt-Winters)
7. Cuantificación de pérdidas de energía
```
