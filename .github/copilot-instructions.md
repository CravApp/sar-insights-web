# Copilot Instructions for AI Coding Agents

## Visión General del Proyecto
- **Propósito:** Monitoreo de degradación forestal en Madre de Dios, Perú, usando datos SAR (Sentinel-1) y algoritmos de IA.
- **Arquitectura:**
  - `app.py`: Aplicación principal Streamlit, integra visualización geoespacial (folium, geopandas), carga de datos raster y vectoriales, y despliegue de resultados de IA.
  - `utils/`: Módulos reutilizables para preprocesamiento (`preprocessing.py`) y modelos ML (`ml_models.py`).
  - `scripts/`: Utilidades para generación de datos sintéticos y pruebas (`generate_sample_data.py`).
  - `config.yaml`: Configuración centralizada de rutas, archivos de entrada y parámetros de visualización.

## Flujos de Trabajo Clave
- **Ejecución local:**
  - Instala dependencias: `pip install -r requirements.txt`
  - Ejecuta la app: `streamlit run app.py`
- **Generación de datos de prueba:**
  - Ejecuta `python scripts/generate_sample_data.py` para poblar `data/` con archivos raster y vector sintéticos.
- **Entrenamiento y uso de modelos:**
  - Los modelos ML se definen en `utils/ml_models.py` (Random Forest). Entrenamiento y predicción se realizan desde la app o scripts auxiliares.

## Convenciones y Patrones
- **Estructura de datos:**
  - Directorios y archivos de datos definidos en `config.yaml` (rutas relativas bajo `data/`).
  - Nombres de archivos raster y vector deben coincidir con los especificados en la sección `input_files` de `config.yaml`.
- **Preprocesamiento:**
  - Usa la clase `SARProcessor` de `utils/preprocessing.py` para cargar y procesar datos SAR.
- **Modelos ML:**
  - Implementa y utiliza la clase `ForestDegradationClassifier` de `utils/ml_models.py`.
- **Visualización:**
  - La app usa Streamlit y folium para mapas interactivos; los estilos y paletas se configuran en `app.py` y `config.yaml`.

## Integraciones y Dependencias
- **Librerías principales:** Streamlit, folium, geopandas, rasterio, scikit-learn, matplotlib.
- **No requiere base de datos ni backend adicional.**
- **Datos de entrada:** Archivos raster (GeoTIFF) y vector (GeoJSON) en `data/`.

## Ejemplo de Flujo Completo
1. Ejecuta `python scripts/generate_sample_data.py` para crear datos.
2. Lanza la app con `streamlit run app.py`.
3. Interactúa con el mapa, visualiza capas y resultados de IA.

## Notas para Agentes
- Mantén la compatibilidad con la estructura de rutas y nombres de archivos definida en `config.yaml`.
- Si agregas nuevas capas, actualiza tanto `config.yaml` como la lógica de carga en `app.py`.
- Sigue los patrones de clase y modularidad de `utils/` para nuevas funcionalidades.
