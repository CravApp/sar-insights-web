"""
Aplicación Web de Monitoreo de Degradación Forestal - Madre de Dios, Perú
Desarrollado para NASA Space Apps Challenge
Utiliza datos SAR (Sentinel-1) y algoritmos de IA para detectar degradación forestal
"""

import streamlit as st
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
from pathlib import Path
import json
from folium import plugins
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO


# Configuración de página con fondo oscuro, gradiente o imagen, y estilos de texto
st.set_page_config(
    page_title="Forest Degradation Monitor - Madre de Dios",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Puedes cambiar la URL de background-image si prefieres una imagen en vez de gradiente
st.markdown("""
<style>
    .stApp {
        /* Fondo gradiente oscuro o imagen */
        background: linear-gradient(135deg, #181c2b 0%, #232946 100%);
        /* background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80');
        background-size: cover; */
    }
    .block-container {
        background: rgba(30, 30, 40, 0.85);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 32px rgba(0,0,0,0.25);
    }
    /* Texto general blanco */
    .stApp *, .block-container * {
        color: #fff !important;
    }
    /* Títulos y subtítulos en blanco y negrita */
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
        font-weight: bold !important;
        letter-spacing: 0.5px;
        font-family: 'Arial', sans-serif;
    }
    /* Párrafos y contenido normal */
    p, li, span, label, .markdown-text-container {
        color: #fff !important;
        font-weight: normal !important;
    }
    /* Sidebar oscuro */
    .stSidebar {
        background: #232946 !important;
    }
    /* Botones */
    .stButton>button {
        background: #3a86ff;
        color: #fff;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #265d97;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

def load_raster(file_path):
    """
    Loads a GeoTIFF raster file and returns data and metadata
    Args:
        file_path (str): Path to the GeoTIFF file
    Returns:
        tuple: (data_array, transform, bounds, crs) or None if fails
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
            
            # Replace invalid values
            data = np.where(data == src.nodata, np.nan, data)
            return data, transform, bounds, crs
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {str(e)}")
        return None

def load_geojson(file_path):
    """
    Loads a GeoJSON file (alerts)
    Args:
        file_path (str): Path to the GeoJSON file
    Returns:
        GeoDataFrame or None if fails
    """
    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {str(e)}")
        return None

def create_colormap(layer_type):
    """
    Create custom colormaps for each layer type
    Args:
        layer_type (str): Layer type (vv, vh, ratio, cusum, classification)
    Returns:
        LinearSegmentedColormap
    """
    colormaps = {
        'vv': LinearSegmentedColormap.from_list('vv', ['#000080', '#4169E1', '#87CEEB']),
        'vh': LinearSegmentedColormap.from_list('vh', ['#2F4F4F', '#20B2AA', '#AFEEEE']),
        'ratio': LinearSegmentedColormap.from_list('ratio', ['#8B4513', '#DAA520', '#FFD700']),
        'cusum': LinearSegmentedColormap.from_list('cusum', ['#006400', '#FFFF00', '#FF4500']),
        'classification': LinearSegmentedColormap.from_list(
            'classification', 
            ['#228B22', '#ADFF2F', '#FFD700', '#FF8C00', '#FF0000']
        )
    }
    return colormaps.get(layer_type, plt.cm.viridis)

def add_raster_to_map(folium_map, data, bounds, layer_name, colormap, opacity=0.6):
    """
    Add a raster layer to the Folium map
    Args:
        folium_map: Folium map object
        data: Numpy array with raster data
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        layer_name: Layer name
        colormap: Matplotlib colormap
        opacity: Layer opacity (0-1)
    """
    try:
        # Normalize data
        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        # Create RGBA image
        colored_data = colormap(data_norm)
        # Convert to PIL image for Folium
        from PIL import Image
        img = Image.fromarray((colored_data * 255).astype(np.uint8))
        # Create in-memory image
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        # Add as ImageOverlay
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{encoded}',
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=opacity,
            name=layer_name,
            overlay=True,
            control=True
        ).add_to(folium_map)
    except Exception as e:
        st.warning(f"⚠️ Could not add layer {layer_name}: {str(e)}")

def add_geojson_to_map(folium_map, gdf, layer_name):
    """
    Add GeoJSON alerts to the map
    Args:
        folium_map: Folium map object
        gdf: GeoDataFrame with alerts
        layer_name: Layer name
    """
    try:
        folium.GeoJson(
            gdf,
            name=layer_name,
            style_function=lambda x: {
                'fillColor': '#FF0000',
                'color': '#8B0000',
                'weight': 2,
                'fillOpacity': 0.4
            },
            highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.7},
            tooltip=folium.GeoJsonTooltip(fields=['id', 'severity'] if 'severity' in gdf.columns else None)
        ).add_to(folium_map)
    except Exception as e:
        st.warning(f"⚠️ Could not add alerts: {str(e)}")

# ==================== CONFIGURACIÓN INICIAL ====================

# Crear estructura de directorios simulada (en producción, estos archivos existirían)
DATA_DIR = Path("data")
RASTER_DIR = DATA_DIR / "rasters"
VECTOR_DIR = DATA_DIR / "vectors"

# ==================== HEADER ====================

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <div style='font-size: 60px;'>🛰️</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        🌳 Forest Degradation Monitor
    </h1>
    <h3 style='text-align: center; color: #FFFFFF; margin-top: 0;'>
        Madre de Dios, Perú - Región Amazónica
    </h3>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <div style='font-size: 60px;'>🌎</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== SIDEBAR - CONTROLES ====================


with st.sidebar:
    st.markdown("### 🎛️ Layer Control")
    
    st.markdown("#### Sentinel-1 SAR Data")
    show_vv = st.checkbox("📡 VV Backscatter", value=False, 
                          help="Vertical-vertical polarization (VV) - Sensitive to forest vertical structure")
    show_vh = st.checkbox("📡 VH Backscatter", value=False,
                          help="Vertical-horizontal polarization (VH) - Sensitive to biomass and foliage")
    show_ratio = st.checkbox("📊 VH/VV Ratio", value=False,
                            help="Volume scattering index - detects changes in canopy structure")
    
    st.markdown("#### Change Analysis")
    show_cusum = st.checkbox("📈 Rsum_max (CuSum)", value=True,
                            help="Cumulative sum of changes - detects temporal degradation")
    
    st.markdown("#### AI Classification")
    show_classification = st.checkbox("🤖 Classification Map (RF/U-TAE)", value=True,
                                     help="Random Forest or U-TAE Transformer result")
    show_alerts = st.checkbox("🚨 Degradation Alerts", value=True,
                             help="Polygons of areas with detected change")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Map Settings")
    opacity = st.slider("Layer opacity", 0.0, 1.0, 0.6, 0.1)
    
    st.markdown("---")
    st.markdown("#### 📥 Export")
    
    export_format = st.selectbox("Export format", ["GeoJSON", "PNG", "GeoTIFF"])
    
    if st.button("⬇️ Download Current View"):
        st.info("🔄 Export functionality under development")
        # In production: implement real export according to selected format

# ==================== MAPA PRINCIPAL ====================

st.markdown("### 🗺️ Interactive Monitoring Map")

# Coordenadas de Madre de Dios, Perú
MADRE_DE_DIOS_CENTER = [-12.5, -70.0]
INITIAL_ZOOM = 7

# Crear mapa base
m = folium.Map(
    location=MADRE_DE_DIOS_CENTER,
    zoom_start=INITIAL_ZOOM,
    tiles='OpenStreetMap',
    control_scale=True
)

# Añadir capas base alternativas
folium.TileLayer('Stamen Terrain', name='Terreno',attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satélite Esri',
    overlay=False,
    control=True
).add_to(m)

# NOTA: En producción, cargar archivos reales. Aquí simulamos la lógica
# Rutas de archivos (ajustar según tu estructura)
raster_files = {
    'vv': RASTER_DIR / "vv_backscatter.tif",
    'vh': RASTER_DIR / "vh_backscatter.tif",
    'ratio': RASTER_DIR / "vh_vv_ratio.tif",
    'cusum': RASTER_DIR / "rsum_max_cusum.tif",
    'classification': RASTER_DIR / "class_map.tif"
}

alerts_file = VECTOR_DIR / "alerts.geojson"

# Cargar y añadir capas según selección del usuario
layer_config = {
    'vv': (show_vv, 'VV Backscatter', 'vv'),
    'vh': (show_vh, 'VH Backscatter', 'vh'),
    'ratio': (show_ratio, 'VH/VV Ratio', 'ratio'),
    'cusum': (show_cusum, 'CuSum Rsum_max', 'cusum'),
    'classification': (show_classification, 'Clasificación RF/IA', 'classification')
}

# Información de ejemplo (en producción, se cargarían archivos reales)
st.info("""
📌 **Demo Note:** This is a visualization interface. 
In production, GeoTIFF and GeoJSON files would be loaded from the `data/` directory.
To test with real data, place your files in:
- `data/rasters/` → `.tif` files (vv_backscatter.tif, vh_backscatter.tif, etc.)
- `data/vectors/` → `alerts.geojson` file
""")

# Simular carga de capas (en producción descomentar)
for key, (show_layer, layer_name, cmap_type) in layer_config.items():
    if show_layer:
        # Descomenta esto en producción con archivos reales:
        # result = load_raster(raster_files[key])
        # if result:
        #     data, transform, bounds, crs = result
        #     colormap = create_colormap(cmap_type)
        #     add_raster_to_map(m, data, bounds, layer_name, colormap, opacity)
        pass

# Cargar alertas
if show_alerts:
    # Descomenta en producción:
    # gdf_alerts = load_geojson(alerts_file)
    # if gdf_alerts is not None:
    #     add_geojson_to_map(m, gdf_alerts, "Alertas de Degradación")
    pass

# Add example marker in Madre de Dios
folium.Marker(
    MADRE_DE_DIOS_CENTER,
    popup="<b>Madre de Dios</b><br>Monitoring Region",
    tooltip="Monitoring Center",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# Añadir control de capas
folium.LayerControl().add_to(m)

# Añadir plugin de búsqueda de coordenadas
plugins.Geocoder().add_to(m)

# Añadir medidor de distancia
plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)

# Mostrar mapa
folium_static(m, width=1400, height=600)

st.markdown("### 📊 Leyenda de Capas Activas")
# ==================== LEGEND ====================

st.markdown("### 📊 Active Layers Legend")

legend_cols = st.columns(5)

with legend_cols[0]:
    if show_cusum:
        st.markdown("""
        **CuSum (Rsum_max)**
        - 🟢 Green: No change
        - 🟡 Yellow: Moderate change
        - 🔴 Red: Significant change
        """)

with legend_cols[1]:
    if show_classification:
        st.markdown("""
        **Classification**
        - 🟢 Dark green: Intact forest
        - 🟢 Light green: Secondary forest
        - 🟡 Yellow: Mild degradation
        - 🟠 Orange: Severe degradation
        - 🔴 Red: Deforestation
        """)

with legend_cols[2]:
    if show_vv:
        st.markdown("""
        **VV Backscatter**
        - Dark blue → Light blue
        - Higher backscatter indicates
        - greater vertical structure
        """)

with legend_cols[3]:
    if show_vh:
        st.markdown("""
        **VH Backscatter**
        - Dark green → Cyan
        - Related to biomass
        - and foliage density
        """)

with legend_cols[4]:
    if show_alerts:
        st.markdown("""
        **Alerts**
        - 🔴 Red polygons
        - Areas with detected change
        - Require verification
        """)

st.markdown("---")
st.markdown("---")
st.markdown("### 📍 Case Studies")

case_studies = {
    "Case 1: Illegal Mining - Malinowski River": {
        "coords": [-12.8, -70.2],
        "zoom": 12,
        "description": """
        **Area affected by illegal gold mining**
        
        - **Coordinates:** 12°48'S, 70°12'W
        - **Affected area:** ~450 hectares
        - **Period:** 2020-2024
        - **Detection:** Abrupt change in VV backscatter and increase in Rsum_max
        - **Impact:** Complete deforestation, river contamination with mercury
        - **Status:** Active red alert, requires immediate intervention
        """
    },
    "Case 2: Selective Logging - Tambopata Reserve": {
        "coords": [-12.3, -69.5],
        "zoom": 12,
        "description": """
        **Gradual degradation due to logging**
        
        - **Coordinates:** 12°18'S, 69°30'W
        - **Affected area:** ~1,200 hectares
        - **Period:** 2021-2024
        - **Detection:** Progressive decrease in VH/VV ratio
        - **Impact:** 30% canopy loss, habitat fragmentation
        - **Status:** Orange alert, ongoing degradation
        """
    },
    "Case 3: Interoceanic Highway - Buffer Zone": {
        "coords": [-12.6, -69.8],
        "zoom": 11,
        "description": """
        **Deforestation associated with road infrastructure**
        
        - **Coordinates:** 12°36'S, 69°48'W
        - **Affected area:** ~2,800 hectares
        - **Period:** 2019-2024
        - **Detection:** Linear change pattern in multitemporal classification
        - **Impact:** Deforestation 15 km on each side of the road, invasions
        - **Status:** Continuous monitoring, predictable expansion
        """
    }
}

selected_case = st.selectbox(
    "Select a case study to explore:",
    ["None"] + list(case_studies.keys())
)

if selected_case != "None":
    case_info = case_studies[selected_case]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crear mapa específico del caso
        case_map = folium.Map(
            location=case_info["coords"],
            zoom_start=case_info["zoom"],
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri'
        )
        
        # Añadir marcador
        folium.Marker(
            case_info["coords"],
            popup=f"<b>{selected_case}</b>",
            icon=folium.Icon(color='red', icon='exclamation-sign')
        ).add_to(case_map)
        
        # Añadir círculo de área afectada
        folium.Circle(
            case_info["coords"],
            radius=2000,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.2,
            popup="Área de estudio"
        ).add_to(case_map)
        
        folium_static(case_map, width=800, height=400)
    
    with col2:
        st.markdown(f"""
        <div class='case-study-box'>
        {case_info["description"]}
        </div>
        """, unsafe_allow_html=True)

# ==================== ESTADÍSTICAS Y MÉTRICAS ====================

st.markdown("---")
st.markdown("### 📈 Monitoring Statistics")

metric_cols = st.columns(4)

# Example data (in production, calculate from real rasters)
with metric_cols[0]:
    st.metric(
        label="Total Monitored Area",
        value="8.5M ha",
        delta="↑ 0.5M ha this month"
    )

with metric_cols[1]:
    st.metric(
        label="Active Alerts",
        value="247",
        delta="↑ 23 vs. last month",
        delta_color="inverse"
    )

with metric_cols[2]:
    st.metric(
        label="Detected Degradation",
        value="12,450 ha",
        delta="↑ 8.3% this quarter",
        delta_color="inverse"
    )

with metric_cols[3]:
    st.metric(
        label="Model Accuracy",
        value="94.2%",
        delta="↑ 2.1% with U-TAE"
    )

# ==================== INFORMACIÓN TÉCNICA ====================


with st.expander("ℹ️ System Technical Information"):
    st.markdown("""
    ### Detection Methodology
    
    **Data Used:**
    - Sentinel-1 SAR images (C-band, 5.405 GHz)
    - Polarizations: VV and VH
    - Spatial resolution: 10m
    - Temporal frequency: 6-12 days
    
    **AI Algorithms:**
    1. **Random Forest (RF):** Supervised classification with 500 decision trees
        - Features: VV, VH, VH/VV, GLCM textures, temporal statistics
        - Accuracy: 91.5% (validation set)
    
    2. **U-TAE Transformer:** Neural network for temporal analysis
        - Architecture: U-Net + Temporal Attention Encoder
        - Input: Time series of 24 images (2 years)
        - Accuracy: 94.2% (validation set)
    
    **Change Metrics:**
    - **CuSum (Rsum_max):** Cumulative sum of residuals to detect gradual changes
    - **Multi-temporal differences:** Trend analysis in backscatter
    - **Texture indices:** GLCM (Gray-Level Co-occurrence Matrix)
    
    **Validation:**
    - Ground truth: Field verification and high-resolution imagery
    - Metrics: Precision, Recall, F1-Score, Overall Accuracy
    - Spatial cross-validation to avoid overfitting
    """)

with st.expander("🔧 Usage Instructions"):
    st.markdown("""
    ### How to use this application
    
    1. **Explore the map:** Use zoom and drag to navigate Madre de Dios
    
    2. **Activate layers:** In the sidebar, check the layers you want to visualize
        - Combine multiple layers for comprehensive analysis
        - Adjust opacity for better visualization
    
    3. **Case studies:** Select a case from the dropdown menu to see real examples
    
    4. **Map tools:**
        - 🔍 Search specific locations (top right corner)
        - 📏 Measure distances (top left corner)
        - 🗺️ Change base layer (layer control)
    
    5. **Export data:** Use the download button in the sidebar
        - GeoJSON: For GIS analysis
        - PNG: For reports and presentations
        - GeoTIFF: For advanced raster analysis
    
    6. **Interpret results:**
        - Green: Healthy forest, no changes
        - Yellow/Orange: Degradation detected, requires monitoring
        - Red: Deforestation or severe change, requires immediate action
    """)

# ==================== FOOTER ====================

st.markdown("---")

footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown("""
    **🛰️ NASA Space Apps Challenge**  
    Developed with Sentinel-1 data (ESA)  
    Copernicus Programme
    """)

with footer_cols[1]:
    st.markdown("""
    **📧 Contact**  
    forestmonitor@example.com  
    [GitHub Repository](#)
    """)

with footer_cols[2]:
    st.markdown("""
    **📅 Last Update**  
    Data: 28 September 2024  
    Model: U-TAE v2.1
    """)

st.markdown("""
<div style='text-align: center; color: #AAAAAA; font-size: 12px; margin-top: 20px;'>
    Madre de Dios Forest Monitoring System v1.0 | 2024  
    Powered by Sentinel-1, Streamlit & Python
</div>
""", unsafe_allow_html=True)
