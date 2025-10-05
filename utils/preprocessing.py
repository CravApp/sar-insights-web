"""
Módulo de Preprocesamiento de Datos SAR
Funciones para cargar, procesar y preparar datos Sentinel-1
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class SARProcessor:
    """
    Clase para procesar datos SAR Sentinel-1
    """
    
    def __init__(self, data_dir: Path):
        """
        Inicializa el procesador SAR
        
        Args:
            data_dir: Directorio raíz con datos raster y vector
        """
        self.data_dir = Path(data_dir)
        self.raster_dir = self.data_dir / "rasters"
        self.vector_dir = self.data_dir / "vectors"
        
    def load_raster_safe(self, 
                        filename: str, 
                        band: int = 1,
                        downscale_factor: float = 1.0) -> Optional[Dict]:
        """
        Carga un archivo raster de forma segura con manejo de errores
        
        Args:
            filename: Nombre del archivo GeoTIFF
            band: Número de banda a cargar (default: 1)
            downscale_factor: Factor de reducción de resolución (1.0 = original)
        
        Returns:
            Diccionario con data, transform, bounds, crs o None si falla
        """
        filepath = self.raster_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Archivo no encontrado: {filepath}")
            return None
            
        try:
            with rasterio.open(filepath) as src:
                # Leer metadata
                original_width = src.width
                original_height = src.height
                
                # Calcular nuevas dimensiones si hay downscaling
                if downscale_factor != 1.0:
                    new_width = int(original_width * downscale_factor)
                    new_height = int(original_height * downscale_factor)
                    
                    # Leer con resampling
                    data = src.read(
                        band,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                    
                    # Ajustar transform para nueva resolución
                    transform = src.transform * src.transform.scale(
                        (original_width / new_width),
                        (original_height / new_height)
                    )
                else:
                    data = src.read(band)
                    transform = src.transform
                
                # Reemplazar nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                return {
                    'data': data,
                    'transform': transform,
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'width': data.shape[1],
                    'height': data.shape[0]
                }
                
        except Exception as e:
            print(f"❌ Error al cargar {filename}: {str(e)}")
            return None
    
    def calculate_ratio(self, 
                       vv_file: str, 
                       vh_file: str,
                       epsilon: float = 1e-10) -> Optional[np.ndarray]:
        """
        Calcula el ratio VH/VV
        
        Args:
            vv_file: Nombre del archivo VV
            vh_file: Nombre del archivo VH
            epsilon: Valor pequeño para evitar división por cero
        
        Returns:
            Array numpy con el ratio o None si falla
        """
        vv_data = self.load_raster_safe(vv_file)
        vh_data = self.load_raster_safe(vh_file)
        
        if vv_data is None or vh_data is None:
            return None
        
        # Convertir a lineal si está en dB
        vv_linear = 10 ** (vv_data['data'] / 10)
        vh_linear = 10 ** (vh_data['data'] / 10)
        
        # Calcular ratio
        ratio = vh_linear / (vv_linear + epsilon)
        
        return ratio
    
    def apply_speckle_filter(self, 
                           data: np.ndarray, 
                           window_size: int = 5) -> np.ndarray:
        """
        Aplica filtro Lee para reducir speckle en datos SAR
        
        Args:
            data: Array numpy con datos SAR
            window_size: Tamaño de la ventana (debe ser impar)
        
        Returns:
            Array numpy filtrado
        """
        from scipy.ndimage import uniform_filter
        
        # Asegurar que window_size es impar
        if window_size % 2 == 0:
            window_size += 1
        
        # Calcular media y varianza local
        mean = uniform_filter(data, size=window_size)
        sqr_mean = uniform_filter(data**2, size=window_size)
        variance = sqr_mean - mean**2
        
        # Varianza del ruido (estimada)
        overall_variance = np.nanvar(data)
        
        # Coeficiente de variación
        weights = variance / (variance + overall_variance + 1e-10)
        
        # Aplicar filtro Lee
        filtered = mean + weights * (data - mean)
        
        return filtered
    
    def calculate_cusum(self, 
                       time_series: np.ndarray, 
                       threshold: float = 0.5) -> np.ndarray:
        """
        Calcula CuSum (Cumulative Sum) para series temporales
        
        Args:
            time_series: Array 3D (time, height, width)
            threshold: Umbral para detección de cambios
        
        Returns:
            Array 2D con Rsum_max (valor máximo de CuSum)
        """
        # Calcular media temporal
        mean_value = np.nanmean(time_series, axis=0)
        
        # Calcular desviación estándar temporal
        std_value = np.nanstd(time_series, axis=0) + 1e-10
        
        # Normalizar serie temporal
        normalized = (time_series - mean_value) / std_value
        
        # Calcular CuSum
        cusum = np.zeros_like(time_series)
        cusum[0] = np.maximum(0, normalized[0] - threshold)
        
        for t in range(1, time_series.shape[0]):
            cusum[t] = np.maximum(0, cusum[t-1] + normalized[t] - threshold)
        
        # Retornar valor máximo (Rsum_max)
        rsum_max = np.nanmax(cusum, axis=0)
        
        return rsum_max
    
    def load_alerts_geojson(self, filename: str = "alerts.geojson") -> Optional[gpd.GeoDataFrame]:
        """
        Carga archivo GeoJSON de alertas
        
        Args:
            filename: Nombre del archivo GeoJSON
        
        Returns:
            GeoDataFrame o None si falla
        """
        filepath = self.vector_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Archivo de alertas no encontrado: {filepath}")
            return None
        
        try:
            gdf = gpd.read_file(filepath)
            
            # Validar que tenga geometrías
            if gdf.empty:
                print("⚠️ El archivo GeoJSON está vacío")
                return None
            
            return gdf
            
        except Exception as e:
            print(f"❌ Error al cargar alertas: {str(e)}")
            return None
    
    def export_to_geotiff(self, 
                         data: np.ndarray,
                         output_path: str,
                         reference_file: str,
                         dtype: str = 'float32') -> bool:
        """
        Exporta array numpy a GeoTIFF
        
        Args:
            data: Array numpy a exportar
            output_path: Ruta del archivo de salida
            reference_file: Archivo de referencia para metadata
            dtype: Tipo de dato (float32, int16, etc.)
        
        Returns:
            True si se exportó exitosamente, False en caso contrario
        """
        ref_filepath = self.raster_dir / reference_file
        
        if not ref_filepath.exists():
            print(f"❌ Archivo de referencia no encontrado: {ref_filepath}")
            return False
        
        try:
            with rasterio.open(ref_filepath) as src:
                # Copiar metadata
                meta = src.meta.copy()
                meta.update({
                    'dtype': dtype,
                    'count': 1,
                    'compress': 'lzw'
                })
                
                # Escribir archivo
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data.astype(dtype), 1)
                
                print(f"✅ Exportado exitosamente: {output_path}")
                return True
                
        except Exception as e:
            print(f"❌ Error al exportar: {str(e)}")
            return False


def calculate_glcm_features(data: np.ndarray, 
                           distances: list = [1], 
                           angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, np.ndarray]:
    """
    Calcula características de textura GLCM (Gray-Level Co-occurrence Matrix)
    
    Args:
        data: Array numpy con datos raster
        distances: Distancias para GLCM
        angles: Ángulos para GLCM (en radianes)
    
    Returns:
        Diccionario con características: contrast, correlation, energy, homogeneity
    """
    from skimage.feature import graycomatrix, graycoprops
    
    # Normalizar a rango 0-255
    data_norm = ((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)) * 255).astype(np.uint8)
    
    # Reemplazar NaN con 0
    data_norm = np.nan_to_num(data_norm, nan=0)
    
    # Calcular GLCM
    glcm = graycomatrix(data_norm, distances=distances, angles=angles, 
                       levels=256, symmetric=True, normed=True)
    
    # Calcular propiedades
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean()
    }
    
    return features


def create_temporal_composite(file_list: list, 
                              statistic: str = 'median') -> np.ndarray:
    """
    Crea un compuesto temporal de múltiples imágenes
    
    Args:
        file_list: Lista de rutas a archivos raster
        statistic: Estadística a calcular ('mean', 'median', 'max', 'min')
    
    Returns:
        Array numpy con el compuesto temporal
    """
    data_stack = []
    
    for file_path in file_list:
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = np.where(data == src.nodata, np.nan, data)
                data_stack.append(data)
        except Exception as e:
            print(f"⚠️ Error al cargar {file_path}: {str(e)}")
            continue
    
    if not data_stack:
        return None
    
    data_stack = np.array(data_stack)
    
    # Calcular estadística
    if statistic == 'mean':
        composite = np.nanmean(data_stack, axis=0)
    elif statistic == 'median':
        composite = np.nanmedian(data_stack, axis=0)
    elif statistic == 'max':
        composite = np.nanmax(data_stack, axis=0)
    elif statistic == 'min':
        composite = np.nanmin(data_stack, axis=0)
    else:
        raise ValueError(f"Estadística no soportada: {statistic}")
    
    return composite


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar procesador
    processor = SARProcessor(Path("data"))
    
    # Cargar datos VV
    vv_data = processor.load_raster_safe("vv_backscatter.tif", downscale_factor=0.5)
    
    if vv_data:
        print(f"✅ VV cargado: {vv_data['width']}x{vv_data['height']} pixels")
        
        # Aplicar filtro de speckle
        filtered = processor.apply_speckle_filter(vv_data['data'])
        print("✅ Filtro de speckle aplicado")
        
        # Calcular características GLCM
        glcm_features = calculate_glcm_features(filtered)
        print(f"✅ GLCM calculado: {glcm_features}")
    
    # Cargar alertas
    alerts = processor.load_alerts_geojson()
    if alerts is not None:
        print(f"✅ Alertas cargadas: {len(alerts)} polígonos")
