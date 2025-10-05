"""
Script para generar datos de prueba sintéticos
Útil para demostración y testing de la aplicación
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import Polygon, Point
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuración
MADRE_DE_DIOS_BOUNDS = (-71.5, -13.5, -68.5, -11.5)  # minx, miny, maxx, maxy
WIDTH = 1000
HEIGHT = 800
OUTPUT_DIR = Path("data")
RASTER_DIR = OUTPUT_DIR / "rasters"
VECTOR_DIR = OUTPUT_DIR / "vectors"

# Crear directorios
RASTER_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_sar_data(width: int, height: int, 
                              data_type: str = 'vv') -> np.ndarray:
    """
    Genera datos SAR sintéticos con patrones realistas
    
    Args:
        width: Ancho del raster
        height: Alto del raster
        data_type: Tipo de dato ('vv', 'vh', 'ratio', 'cusum')
    
    Returns:
        Array numpy con datos sintéticos
    """
    np.random.seed(42)
    
    # Crear base
    if data_type == 'vv':
        # VV backscatter: -20 a -5 dB
        base = np.random.uniform(-20, -5, (height, width))
        # Añadir zonas de degradación (valores más bajos)
        degradation_mask = create_degradation_zones(width, height, num_zones=3)
        base[degradation_mask] -= 5
        
    elif data_type == 'vh':
        # VH backscatter: -25 a -10 dB
        base = np.random.uniform(-25, -10, (height, width))
        degradation_mask = create_degradation_zones(width, height, num_zones=3)
        base[degradation_mask] -= 5
        
    elif data_type == 'ratio':
        # VH/VV ratio: 0.1 a 0.9
        base = np.random.uniform(0.1, 0.9, (height, width))
        degradation_mask = create_degradation_zones(width, height, num_zones=3)
        base[degradation_mask] *= 0.5  # Menor ratio en áreas degradadas
        
    elif data_type == 'cusum':
        # CuSum: 0 a 10
        base = np.random.uniform(0, 2, (height, width))
        degradation_mask = create_degradation_zones(width, height, num_zones=3)
        base[degradation_mask] += np.random.uniform(5, 10, degradation_mask.sum())
    
    else:
        raise ValueError(f"Tipo de dato no soportado: {data_type}")
    
    # Añadir ruido speckle realista
    noise = np.random.normal(0, 0.5, (height, width))
    base += noise
    
    # Suavizar con filtro gaussiano simple
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=2)
    
    return base.astype(np.float32)


def create_degradation_zones(width: int, height: int, num_zones: int = 3) -> np.ndarray:
    """
    Crea zonas de degradación sintéticas con formas realistas
    
    Args:
        width: Ancho del array
        height: Alto del array
        num_zones: Número de zonas de degradación
    
    Returns:
        Array booleano con zonas de degradación
    """
    mask = np.zeros((height, width), dtype=bool)
    
    for _ in range(num_zones):
        # Centro aleatorio
        center_y = np.random.randint(height // 4, 3 * height // 4)
        center_x = np.random.randint(width // 4, 3 * width // 4)
        
        # Tamaño aleatorio
        radius_y = np.random.randint(30, 80)
        radius_x = np.random.randint(30, 80)
        
        # Crear elipse
        y, x = np.ogrid[:height, :width]
        zone = ((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2 <= 1
        mask |= zone
    
    return mask


def create_classification_map(cusum: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """
    Genera mapa de clasificación basado en CuSum y VV
    
    Args:
        cusum: Array con valores de CuSum
        vv: Array con valores VV
    
    Returns:
        Array con clases (0-4)
    """
    classification = np.zeros_like(cusum, dtype=np.int8)
    
    # Clase 0: Bosque Intacto (VV alto, CuSum bajo)
    classification[(vv > -10) & (cusum < 2)] = 0
    
    # Clase 1: Bosque Secundario
    classification[(vv > -12) & (vv <= -10) & (cusum >= 2) & (cusum < 4)] = 1
    
    # Clase 2: Degradación Leve
    classification[(cusum >= 4) & (cusum < 6)] = 2
    
    # Clase 3: Degradación Severa
    classification[(cusum >= 6) & (cusum < 8)] = 3
    
    # Clase 4: Deforestación
    classification[cusum >= 8] = 4
    
    return classification


def save_geotiff(data: np.ndarray, filename: str, bounds: tuple, crs: str = 'EPSG:4326'):
    """
    Guarda array numpy como GeoTIFF
    
    Args:
        data: Array a guardar
        filename: Nombre del archivo de salida
        bounds: (minx, miny, maxx, maxy)
        crs: Sistema de coordenadas
    """
    transform = from_bounds(*bounds, data.shape[1], data.shape[0])
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS.from_string(crs),
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
    
    print(f"✅ Guardado: {filename}")


def generate_alerts_geojson(classification: np.ndarray, 
                           bounds: tuple,
                           threshold: int = 2) -> gpd.GeoDataFrame:
    """
    Genera alertas GeoJSON basadas en clasificación
    
    Args:
        classification: Array con clases
        bounds: (minx, miny, maxx, maxy)
        threshold: Clase mínima para generar alerta (2 = degradación leve)
    
    Returns:
        GeoDataFrame con polígonos de alertas
    """
    from scipy import ndimage
    
    # Detectar zonas con degradación
    alert_mask = classification >= threshold
    
    # Etiquetar regiones conectadas
    labeled_array, num_features = ndimage.label(alert_mask)
    
    # Crear polígonos para cada región
    polygons = []
    severities = []
    areas = []
    ids = []
    
    minx, miny, maxx, maxy = bounds
    pixel_width = (maxx - minx) / classification.shape[1]
    pixel_height = (maxy - miny) / classification.shape[0]
    
    for region_id in range(1, num_features + 1):
        # Obtener píxeles de la región
        region_mask = labeled_array == region_id
        region_pixels = np.argwhere(region_mask)
        
        # Calcular área (en hectáreas aproximadamente)
        area_ha = len(region_pixels) * abs(pixel_width * pixel_height) * 111000 * 111000 / 10000
        
        # Filtrar regiones muy pequeñas
        if area_ha < 5:
            continue
        
        # Calcular severidad promedio
        severity_values = classification[region_mask]
        avg_severity = int(np.mean(severity_values))
        
        # Crear polígono simple (bounding box de la región)
        rows, cols = region_pixels[:, 0], region_pixels[:, 1]
        
        # Convertir a coordenadas geográficas
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        lon_min = minx + min_col * pixel_width
        lon_max = minx + (max_col + 1) * pixel_width
        lat_min = maxy - (max_row + 1) * pixel_height
        lat_max = maxy - min_row * pixel_height
        
        # Crear polígono
        poly = Polygon([
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
            (lon_min, lat_min)
        ])
        
        polygons.append(poly)
        severities.append(avg_severity)
        areas.append(round(area_ha, 2))
        ids.append(f"ALERT_{region_id:03d}")
    
    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'id': ids,
        'severity': severities,
        'area_ha': areas,
        'status': ['active'] * len(ids),
        'confidence': np.random.uniform(0.75, 0.98, len(ids)).round(2)
    }, geometry=polygons, crs='EPSG:4326')
    
    return gdf


def main():
    """
    Función principal para generar todos los datos de prueba
    """
    print("🚀 Generando datos sintéticos para Madre de Dios, Perú\n")
    
    # 1. Generar datos SAR
    print("1️⃣ Generando datos SAR Sentinel-1...")
    
    vv_data = create_synthetic_sar_data(WIDTH, HEIGHT, 'vv')
    save_geotiff(vv_data, RASTER_DIR / "vv_backscatter.tif", MADRE_DE_DIOS_BOUNDS)
    
    vh_data = create_synthetic_sar_data(WIDTH, HEIGHT, 'vh')
    save_geotiff(vh_data, RASTER_DIR / "vh_backscatter.tif", MADRE_DE_DIOS_BOUNDS)
    
    ratio_data = create_synthetic_sar_data(WIDTH, HEIGHT, 'ratio')
    save_geotiff(ratio_data, RASTER_DIR / "vh_vv_ratio.tif", MADRE_DE_DIOS_BOUNDS)
    
    cusum_data = create_synthetic_sar_data(WIDTH, HEIGHT, 'cusum')
    save_geotiff(cusum_data, RASTER_DIR / "rsum_max_cusum.tif", MADRE_DE_DIOS_BOUNDS)
    
    print()
    
    # 2. Generar clasificación
    print("2️⃣ Generando mapa de clasificación...")
    classification = create_classification_map(cusum_data, vv_data)
    save_geotiff(classification, RASTER_DIR / "class_map.tif", MADRE_DE_DIOS_BOUNDS)
    
    # Estadísticas de clasificación
    unique, counts = np.unique(classification, return_counts=True)
    class_names = {
        0: 'Bosque Intacto',
        1: 'Bosque Secundario',
        2: 'Degradación Leve',
        3: 'Degradación Severa',
        4: 'Deforestación'
    }
    
    print("\nDistribución de clases:")
    for cls, count in zip(unique, counts):
        percentage = (count / classification.size) * 100
        print(f"   {class_names.get(cls, 'Unknown')}: {percentage:.1f}% ({count} píxeles)")
    
    print()
    
    # 3. Generar alertas GeoJSON
    print("3️⃣ Generando alertas de degradación...")
    alerts_gdf = generate_alerts_geojson(classification, MADRE_DE_DIOS_BOUNDS, threshold=2)
    alerts_gdf.to_file(VECTOR_DIR / "alerts.geojson", driver='GeoJSON')
    print(f"✅ Guardado: {VECTOR_DIR / 'alerts.geojson'}")
    print(f"   Total de alertas: {len(alerts_gdf)}")
    print(f"   Área total afectada: {alerts_gdf['area_ha'].sum():.2f} hectáreas")
    
    print()
    
    # 4. Crear metadata JSON
    print("4️⃣ Generando metadata...")
    metadata = {
        "dataset": "Synthetic SAR Data - Madre de Dios",
        "region": "Madre de Dios, Peru",
        "bounds": {
            "minx": MADRE_DE_DIOS_BOUNDS[0],
            "miny": MADRE_DE_DIOS_BOUNDS[1],
            "maxx": MADRE_DE_DIOS_BOUNDS[2],
            "maxy": MADRE_DE_DIOS_BOUNDS[3]
        },
        "dimensions": {
            "width": WIDTH,
            "height": HEIGHT
        },
        "crs": "EPSG:4326",
        "sensor": "Sentinel-1 (simulated)",
        "date_generated": "2024-09-28",
        "files": {
            "vv_backscatter": "vv_backscatter.tif",
            "vh_backscatter": "vh_backscatter.tif",
            "vh_vv_ratio": "vh_vv_ratio.tif",
            "cusum": "rsum_max_cusum.tif",
            "classification": "class_map.tif",
            "alerts": "alerts.geojson"
        },
        "statistics": {
            "total_alerts": len(alerts_gdf),
            "total_area_ha": float(alerts_gdf['area_ha'].sum()),
            "class_distribution": {
                class_names[cls]: int(count) for cls, count in zip(unique, counts)
            }
        }
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Guardado: {OUTPUT_DIR / 'metadata.json'}")
    
    print("\n" + "="*60)
    print("✅ GENERACIÓN DE DATOS COMPLETADA")
    print("="*60)
    print(f"\n📁 Los datos están disponibles en: {OUTPUT_DIR.absolute()}")
    print("\n🚀 Para ejecutar la aplicación:")
    print("   streamlit run app.py")
    print("\n📊 Archivos generados:")
    print(f"   - Rasters: {len(list(RASTER_DIR.glob('*.tif')))} archivos GeoTIFF")
    print(f"   - Vectores: {len(list(VECTOR_DIR.glob('*.geojson')))} archivos GeoJSON")
    print(f"   - Metadata: metadata.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error durante la generación: {str(e)}")
        import traceback
        traceback.print_exc()
