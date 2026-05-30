
import numpy as np

def overlay_back_roi(roi_vol, bbox, fill_value=0.0):
    """
    Inserta un ROI (Droi,Hroi,Wroi) dentro de un volumen negro del tamaño original.
    roi_vol: ndarray (Droi,Hroi,Wroi) o (Droi,Hroi,Wroi,1)
    bbox: (minimos, maximos, siz) devuelto por enmascarar(return_bbox=True)
    """
    minimos, maximos, siz = bbox

    # siz puede ser (N,D,H,W) si venía de 4D; queremos (D,H,W)
    if len(siz) == 4:
        D0, H0, W0 = siz[1], siz[2], siz[3]
    else:
        D0, H0, W0 = siz[0], siz[1], siz[2]

    # quitamos canal si viene (D,H,W,1)
    if roi_vol.ndim == 4 and roi_vol.shape[-1] == 1:
        roi_vol = roi_vol[..., 0]

    out = np.full((D0, H0, W0), fill_value, dtype=roi_vol.dtype)

    out[minimos[0]:maximos[0]+1,
        minimos[1]:maximos[1]+1,
        minimos[2]:maximos[2]+1] = roi_vol

    return out

def normalize_clip01(x, p_low=1, p_high=99, return_params=False):
    lo, hi = np.percentile(x, (p_low, p_high))
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    x = x.astype(np.float32)

    if return_params:
        return x, float(lo), float(hi)
    return x

def denormalize_from_01(x01, lo, hi):
    """
    x01: ndarray en [0,1]
    lo, hi: escala original
    """
    return x01 * (hi - lo) + lo

def pad_to_multiples(x, multiples=(8,8,8)):
    """
    Acolcha volúmenes 3D a múltiplos (p.ej. 8,8,8) para poder
    aplicar 3 MaxPooling3D (/2 x3) y reconstruir limpiamente.
    Devuelve array acolchado y slices para recortar de vuelta.
    """
    #Esta linea se queda con el volumen
    N, D, H, W = x.shape
    
    #En estas lineas dividimos entre 8, redondeamos hacia arriba y
    #multiplicamos por 8 para poder tener el multiplo de 8 mas cercano
    #Se hace padding por 8 porque hacemos maxpooling 3 veces entre 2
    # np.ceil() redondea hacia arriba los valores
    md, mh, mw = multiples
    D2 = int(np.ceil(D/md) * md)
    H2 = int(np.ceil(H/mh) * mh)
    W2 = int(np.ceil(W/mw) * mw)

    #Calculamos cuanto hace falta introducir de padding
    pd0 = D2 - D; ph0 = H2 - H; pw0 = W2 - W
    pd = (pd0//2, pd0 - pd0//2)
    ph = (ph0//2, ph0 - ph0//2)
    pw = (pw0//2, pw0 - pw0//2)

    #Con np.pad le añadimos padding a x, vemos que la tupla correspondiente al
    #primer vector no se añade nada y a los otros 3 correspondientes al 
    #volumen se les añade el padding izq y der ya calculado. mode="constant" 
    #significa que rellena con ceros
    x2 = np.pad(x, ((0,0), pd, ph, pw), mode="constant")
    
    #crop_slices lo usaremos luego para recuperar la funcion original
    crop_slices = (
        slice(pd[0], pd[0]+D),
        slice(ph[0], ph[0]+H),
        slice(pw[0], pw[0]+W),
    )
    
    #Devuelve el objeto con padding y "el molde" para recortar padding al final
    return x2, crop_slices
