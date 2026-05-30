
import numpy as np
from skimage import measure

def enmascarar(imagen, percentage=0.6, return_bbox=False):
    """
    Enmascara las componentes de una imagen 3D.
    
    Parámetros:
    -----------
    imagen : ndarray
        Imagen 3D o 4D a enmascarar
    percentage : float
        Porcentaje para calcular el umbral (entre 0 y 1)
    
    Retorna:
    --------
    Imask : ndarray
        Imagen recortada a la región de interés
    v : ndarray
        Vértices de la isosuperficie
    """
    """
    Si no enmascaramos la figura para solamente quedarnos con el area de interes
    estaremos midiendo la diferencia entre las medias de intensidad en todo el 
    cerebro. Los portadores de dopamina se encuentran en una region muy concreta
    del cerebro por lo que si tenemos en cuenta el cerebro entero la media se va
    a ver afectada por una gran parte del cerebro que no es de interes y obtendremos
    valores con ruido.
    """
    siz = imagen.shape
    
    # Calcular imagen promedio y quita las dimensiones de tamaño 1
    if siz[0] == 1: 
        Imean = np.squeeze(imagen.astype(float)) 
    else:
        Imean = np.squeeze(np.mean(imagen.astype(float), axis=0)) 
    
    # Calcular umbral
    th = ((np.max(Imean) - np.min(Imean)) * percentage) + np.min(Imean)
    
    # Extraer isosuperficie
    verts, faces, normals, values = measure.marching_cubes(Imean, th)
    v = verts
    
    # Calcular límites mínimos y máximos
    minimos = np.ceil(np.min(v, axis=0)).astype(int)
    maximos = np.floor(np.max(v, axis=0)).astype(int)
    # Recortar imagen
    
    #Obligamos a que la distancia entre minimo[0] y maximo[0] sea par para poder 
    #hacer la simetria 
    if((maximos[0]+minimos[0])%2 == 0):
        maximos[0] = maximos[0]+1
        
    #Recupera la forma del stack y vuelve a añadir los label (entramos por el else)
    if len(siz) == 3:
        Imask = imagen[minimos[0]:maximos[0]+1, 
                      minimos[1]:maximos[1]+1, 
                      minimos[2]:maximos[2]+1]
    else:  # imagen 4D
        Imask = imagen[:, 
                      minimos[0]:maximos[0]+1, 
                      minimos[1]:maximos[1]+1, 
                      minimos[2]:maximos[2]+1]
    if return_bbox:
        bbox = (minimos, maximos, siz)  # siz es shape original (D,H,W) o (N,D,H,W)
        return Imask, v, bbox
    
    return Imask, v