#==============================================================================
#
#***************************   LIBRERIAS   ************************************
#
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pda

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import umap.umap_ as umap

from skimage import measure

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
# ============================================================
#  IMPORTANTE: en PyTorch el formato de tensores es (N, C, D,H,W)
#  mientras que en TensorFlow era (N, D,H,W,C). Lo tendremos en
#  cuenta al convertir los datos a tensores.
# ============================================================


#==============================================================================
#
#***************************     DATOS     ************************************
#
#==============================================================================

data = np.load("stack_int_all_1430.npz", allow_pickle=True)
labels = data["labels"]
X = data["stack_int"]
print("Forma:", X.shape)
print("Tipo:", X.dtype)


#==============================================================================
#
#***************************     CONSTANTES     *******************************
#
#==============================================================================

#Nota: A mayor beta, mas castigo KL_loss u por tanto el modelo forzara a KL_loss 
#para que sea menor

EPOCHS = 600
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LATENT_DIM = 16
BETA_START = 0.0001
BETA_END = 0.00005

#==============================================================================
#
#*****************     REPOSITORIO DE PESOS GUARDADOS     *********************
#
#==============================================================================

#1
#PESOS = "mask_300Epochs_8BS_1e3LR_8LD_0_0001BStart_0_00005BEnd.pt" 

#2
#PESOS = "mask_300Epochs_8BS_1e3LR_8LD_0_001BStart_0_0005BEnd.pt" 

#3
#PESOS = "mask_300Epochs_8BS_1e3LR_4LD_0_0001BStart_0_00005BEnd.pt" 

#4
#PESOS = "mask_300Epochs_8BS_1e3LR_4LD_0_001BStart_0_0005BEnd.pt" 

#5
#PESOS = "mask_300Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt" 

#6
#PESOS = "mask_300Epochs_8BS_1e3LR_16LD_0_001BStart_0_0005BEnd.pt" 

#7
PESOS = "mask_600Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt" 
#==============================================================================


#==============================================================================
#
#***************************   FUNCIONES   ************************************
#
#==============================================================================
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
    
    # Calcular imagen promedio
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

def predict_one_by_loo(X_2d, y, idx, model=None):
    """
    Predice la clase de la muestra idx usando entrenamiento en todas menos idx (LOO).
    X_2d: (N,2) embedding 2D (mu_umap o z_umap)
    y: (N,) etiquetas 0/1 (0=Control, 1=PD)
    idx: índice de la muestra a evaluar
    model: clasificador sklearn (si None -> LogisticRegression)
    """
    X_2d = np.asarray(X_2d)
    y = np.asarray(y).astype(int)

    if model is None:
        model = LogisticRegression(max_iter=2000)

    mask = np.ones(len(y), dtype=bool)
    mask[idx] = False

    X_train, y_train = X_2d[mask], y[mask]
    X_test = X_2d[idx:idx+1]
    y_true = y[idx]

    model.fit(X_train, y_train)
    y_pred = int(model.predict(X_test)[0])

    # Probabilidad (si está disponible)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = float(model.predict_proba(X_test)[0, 1])  # P(PD)

    return y_true, y_pred, y_proba, model

def plot_one_prediction(X_2d, y, idx, y_true, y_pred, title="Clasificación por embedding 2D"):
    X_2d = np.asarray(X_2d)
    y = np.asarray(y).astype(int)

    plt.figure(figsize=(6,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, s=10, cmap="coolwarm", alpha=0.6)

    # Resaltar la muestra
    plt.scatter(X_2d[idx,0], X_2d[idx,1], s=120, facecolors='none', edgecolors='k', linewidths=2)

    true_name = "PD" if y_true == 1 else "Control"
    pred_name = "PD" if y_pred == 1 else "Control"
    ok = "BIEN" if y_true == y_pred else "MAL"

    plt.title(f"{title}\nidx={idx} | true={true_name} | pred={pred_name} {ok}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

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


def cluster_metrics(X_emb, y, name=""):
    """
    X_emb: ndarray (N, d) embeddings (p.ej., mu_all_std, z_all_std, mu_umap...)
    y: etiquetas (N,) en int o str
    Retorna: dict con silhouette y davies_bouldin.
    """
    y = np.asarray(y)
    X_emb = np.asarray(X_emb)

    # Silhouette requiere al menos 2 clusters y que ningún cluster tenga 1 solo punto.
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2:
        return {"name": name, "silhouette": np.nan, "davies_bouldin": np.nan, "note": "Solo 1 clase"}
    if np.any(counts < 2):
        return {"name": name, "silhouette": np.nan, "davies_bouldin": np.nan, "note": "Alguna clase con <2 muestras"}

    sil = silhouette_score(X_emb, y, metric="euclidean")
    db  = davies_bouldin_score(X_emb, y)

    return {"name": name, "silhouette": float(sil), "davies_bouldin": float(db), "note": ""}


def eval_weights_cluster_metrics(pesos_path, x_tensor, labels_num, device, input_shape, batch_size=8):
    vae, latent_dim_ckpt = build_vae_from_checkpoint(pesos_path, device, input_shape)

    mu_list, z_list = [], []
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor[i:i+batch_size].to(device)
            mu_b, logvar_b, z_b = vae.encode(x_batch)
            mu_list.append(mu_b.cpu().numpy())
            z_list.append(z_b.cpu().numpy())

    z_all  = np.concatenate(z_list, axis=0)

    z_std  = StandardScaler().fit_transform(z_all)

    base = os.path.basename(pesos_path)
    out = []
    out.append(cluster_metrics(z_std,  labels_num, name=f"{base} | LD={latent_dim_ckpt} | z_std"))
    return out

def plot_cluster_metrics(df):

    df_plot = df.sort_values("davies_bouldin")

    names = df_plot["name"].values
    sil = df_plot["silhouette"].values
    db  = df_plot["davies_bouldin"].values

    x = np.arange(len(names))

    plt.figure("Cluster Metrics (z_std only)", figsize=(10,5))

    plt.subplot(1,2,1)
    plt.bar(x, sil)
    plt.xticks(x, names)
    plt.title("Silhouette (z_std)")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.bar(x, db)
    plt.xticks(x, names)
    plt.title("Davies-Bouldin (z_std)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
"""
    Esta capa sera la que le de los valores correspondientes a mu y sigma, 
    nuestros parametros para reparametrizacion del VAE.
    
    La clase Sampling es una capa que hereda de Layer, es decir que hereda las
    propiedades de tf.keras.layers.Layer
    Internamente se llama a la funcion call
    
    eps genera ruido gaussiano
    
    esto hara que z dependa de mu y de sigma y por tanto que sea derivable 
    respecto a dichas variables y por tanto pueda calcular gradientes y pueda 
    haber aprendizaje.
    
    Cuando hablamos de aprender, hablamos de cambiar el valor de los pesos, el
    gradiente es una medida que indica si al cambiar un peso la perdia sube o 
    baja
    gradiente positivo → debes bajar el peso

    gradiente negativo → debes subir el peso

    gradiente grande → cambio grande

    gradiente pequeño → cambio pequeño
    
    Para hacer backpropagation necesitamos que haya gradientes y si alguna 
    operacion en la red no es derivable los gradientes se cortan y no hay aprendizaje
    
"""
class Sampling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps


"""
    EN PyTorch vamos a implementar explícitamente las redes como
    clases nn.Module: un encoder 3D, un decoder 3D y el VAE que los
    combina, manteniendo la misma lógica que en la versión Keras.
"""

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Entradas en PyTorch: (N, 1, 48,48,48)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)  # 48 -> 24
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)  # 24 -> 12
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)  # 12 -> 6

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        return x  # (N, 64, 6,6,6)


class Decoder3D(nn.Module):
    """
        AHORA HAY UN CAMBIO SUSTANCIAL EN LA FUNCION DEL DECODER
        
        ANTES
        Entra un tensor y devolvia un tensor
        
        AHORA
        Entra un tensor y devueve un modelo
    """
    def __init__(self, latent_dim, feat_shape):
        """
            Antes de pasar con la explicacion de la reconstruccion de la forma cubica
            vamos a explicar lo que recibe como input el decoder
            
            El VAE recibe un vector latente, por tanto con dec_inp le estamos diciendo
            que cuando se llame a la funcion  crearemos el modelo decoder y su 
            entrada será un vector de tamaño latent_dim, que es el tamaño del espacio latente.
            
            Pretendemos recuperar la forma cubica:
            -> Con tf.math.reduce_prod obtenemos el valor de 8*8*8*64 que sera el
            numero de neuronas de la Dense de proyeccion
            
            Con la funcion Dense tomamos el vector latente (procedente del cuello de
            botella) y lo mapea en un vector del valor de units = 8*8*8*64
            
            Con Reshape pasamos de un vector 1D a un cubo 3D con las dimensiones 
            especificadas en feat_shape
            
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_shape = feat_shape  # (C, D, H, W)
        units = int(np.prod(feat_shape))  # 64*6*6*6
        self.proj = nn.Linear(latent_dim, units)
        # reshape a (N, 64, 6,6,6) en forward

        """
            Proceso de deconvolucion ya explicado en training 1
        """
        # ConvTranspose3d para duplicar tamaño: usamos stride=2 y output_padding=1
        self.dec_up1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 6 → 12
        self.dec_up2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 12 → 24
        self.dec_up3 = nn.ConvTranspose3d(16, 8,  kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  # 24 → 48
        self.output = nn.Conv3d(8, 1, kernel_size=1, padding=0)         # (48,48,48)

    def forward(self, z):
        N = z.size(0)
        x = F.relu(self.proj(z))
        x = x.view(N, *self.feat_shape)  # → (N, 64, 6,6,6)
        x = F.relu(self.dec_up1(x))
        x = F.relu(self.dec_up2(x))
        x = F.relu(self.dec_up3(x))
        x = torch.sigmoid(self.output(x))
        return x  # (N, 1, 48,48,48)


"""
    EN un VAE tenemos 3 modelos:
        -> El encoder
        -> El decoder
        -> EL modelo grande VAE que los junta y gestiona la perdida
    En el anterior caso teniamos solo un modelo que unia encoder y decoder
    
"""

class VAE(nn.Module):
    def __init__(self, latent_dim, beta=1.0, input_shape=(16,8,8)):
        super().__init__()
        self.beta = float(beta)
        self.input_shape = tuple(input_shape)

        self.encoder_net = Encoder3D()

        # dummy con la forma real de entrada
        dummy = torch.zeros(1, 1, *self.input_shape)
        with torch.no_grad():
            feats = self.encoder_net(dummy)

        self.feat_shape = feats.shape[1:]
        flat_dim = int(np.prod(self.feat_shape))

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.sampling = Sampling()
        self.decoder_net = Decoder3D(latent_dim, self.feat_shape)

    def encode(self, x):
        encoder_feats = self.encoder_net(x)  # (N,64,6,6,6)
        flat = encoder_feats.reshape(x.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.sampling(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        # Definimos la pasada "normal" del modelo: x -> z -> x_recon
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    """
        Esta función es el equivalente a definir la función de pérdidas
        dentro de la clase VAE, como hacíamos en train_step de Keras.
        Calcula:
            - recon_loss (MSE)
            - kl_loss (divergencia KL)
            - total_loss = recon_loss + beta * kl_loss
    """
    def compute_loss(self, x):
        x_recon, mu, logvar, z = self.forward(x)

        # 1) Pérdida de reconstrucción (MSE voxel a voxel)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # 2) KL divergence entre q(z|x) y N(0,1)
        kl_div_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        kl_div_loss = torch.mean(kl_div_loss)
        
        #Falta pasar la y que son las etiquetas, esto se llama conditional_vae
        #landa = 0.5 
        #cls_loss = F.cross_entropy(z, y, reduction = 'mean')

        total_loss = recon_loss + self.beta * kl_div_loss# + landa*cls_loss
        return total_loss, recon_loss, kl_div_loss

    """
        Esta función es el equivalente más parecido a model.fit(...)
        en Keras. Integra el bucle de entrenamiento dentro de la clase
        VAE, usando:
            - dataloader para iterar sobre los datos
            - optimizer para actualizar pesos
            - beta_sched para actualizar beta por época
            - lógica de "ModelCheckpoint" y "EarlyStopping"
    """
    def fit(self, dataloader, optimizer, beta_sched,
            epochs, pesos_path, patience, dataset_size, device):

        best_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            beta_sched.on_epoch_begin(epoch)
            running_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for (batch_x,) in progress_bar:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()

                total_loss, recon_loss, kl_div_loss = self.compute_loss(batch_x)

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item() * batch_x.size(0)

                # Mostrar info en la barra (similar al "loss=..., kl_loss=..." de Keras)
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_div_loss.item():.4f}",
                    "beta": f"{self.beta:.6f}",
                    })

            epoch_loss = running_loss / dataset_size
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}  "
                  f"(recon: {recon_loss.item():.6f}  KL: {kl_div_loss.item():.6f})")

            # "ModelCheckpoint"
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), pesos_path)
                print(f"Mejora en loss, guardando pesos en {pesos_path}")
            else:
                epochs_no_improve += 1

            # "EarlyStopping"
            if epochs_no_improve > patience:
                print("Early stopping por paciencia alcanzada.")
                break


"""
    Con esta clase vamos a crear una callback para hacer una beta scheduler
    como ya hemos hecho anteriormente, vamos a heredar la estructura de una 
    callback y vamos a personalizar algunas de sus funciones.
    
    Esto lo haremos porque queremos que el modelo aprenda primero a reconstruir
    y para ello necesitamos una beta baja para que el KL tenga poca importancia,
    cuando han pasado algunas epocas queremos que el modelo aprenda a generalizar
    y organizar el espacio latente, para ello KL debe ir cobrando mas importancia, 
    para ello debemos ir subiendo el valor de beta
    
    -> __init__ importa todos las variables y los guarda como atributos 
    
    -> on_epoch_begin es una funcion heredada de Callback que Keras llama 
    automáticamente al inicio de cada época que permite ir actualizando valores.
    El procedimiento es el de normalizar por numero de epocas para que el 
    crecimiento o decrecimiento sea lineal conforme a las epocas, interpolacion
    lineal
         Si beta_start < beta_end → beta irá subiendo con las épocas.
         Si beta_start > beta_end → beta irá bajando con las épocas.
"""
class BetaScheduler:
    def __init__(self, vae, beta_start, beta_end, n_epochs):
        self.vae = vae
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch):
        # epoch empieza en 0
        t = epoch / max(self.n_epochs - 1, 1)  # normalizar a [0,1]
        # interpolación lineal: start -> end
        new_beta = self.beta_start + (self.beta_end - self.beta_start) * t
        self.vae.beta = float(new_beta)
        print(f"\n[BetaScheduler] Epoch {epoch+1}: beta = {float(new_beta):.6f}")

def build_vae_from_checkpoint(pesos_path, device, input_shape):
    sd = torch.load(pesos_path, map_location=device, weights_only=True)

    # Inferimos latent_dim desde fc_mu.bias (tamaño = latent_dim)
    latent_dim_ckpt = sd["fc_mu.bias"].numel()

    vae = VAE(latent_dim_ckpt, input_shape=input_shape).to(device)
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    return vae, latent_dim_ckpt
#==============================================================================
#
#****************************     MAIN     ************************************
#
#==============================================================================

X_enm, _, bbox = enmascarar(X, return_bbox=True)
print("Forma X original:", X.shape)        # antes de enmascarar
print("Forma X_enm:", X_enm.shape)         # después de enmascarar (esto debería ser N,D,H,W o D,H,W)

x_norm, norm_lo, norm_hi = normalize_clip01(X_enm, return_params=True)
print("Norm params: lo =", norm_lo, " hi =", norm_hi)
print("Forma x_norm:", x_norm.shape)       # mismo que X_enm

X_pad, crop = pad_to_multiples(x_norm)
print("Forma X_pad (sin canal):", X_pad.shape)  # (N,D,H,W)

#En esta linea le añadimos el canal
X_pad = X_pad[..., np.newaxis].astype(np.float32)  # (N, D, H, W, 1)
print("Forma X_pad (con canal):", X_pad.shape)  # (N,D,H,W,1)

"X_pad es nuestos datos de entrada"
# En PyTorch no usamos tf.keras.Input, sino tensores directamente.
# El equivalente será un tensor de forma (N, 1, 48,48,48)
# y definiremos la arquitectura en clases nn.Module.
# inp = tf.keras.Input(shape=(48,48,48,1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Creamos el VAE a partir del encoder y decoder que ya definiste
input_shape = (X_pad.shape[1], X_pad.shape[2], X_pad.shape[3])  # (16,8,8)
vae = VAE(LATENT_DIM, input_shape=input_shape).to(device)

# Datos a tensores PyTorch: (N, D,H,W,1) -> (N,1,D,H,W)
x_dummy_np = X_pad  # (N,48,48,48,1)
x_tensor = torch.from_numpy(np.transpose(x_dummy_np, (0,4,1,2,3)))  # (N,1,48,48,48)
print("Forma x_tensor (PyTorch):", x_tensor.shape)  # (N,1,D,H,W)
dataset = TensorDataset(x_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

beta_sched = BetaScheduler(vae,beta_start=BETA_START, beta_end=BETA_END,n_epochs=EPOCHS)

# "EarlyStopping" y "ModelCheckpoint" manuales
patience = 75

if os.path.exists(PESOS):
    print("Se han encontrado los pesos guardados de un entrenamiento previo...")
    vae.load_state_dict(torch.load(PESOS, map_location=device))
else:
    vae.fit(
        dataloader=dataloader,
        optimizer=optimizer,
        beta_sched=beta_sched,
        epochs=EPOCHS,
        pesos_path=PESOS,
        patience=patience,
        dataset_size=len(dataset),
        device=device,
    )
    # Cargamos los mejores pesos guardados
    vae.load_state_dict(torch.load(PESOS, map_location=device))


# ==========================
# Reconstrucción de ejemplos
# ==========================
def show_full_pair(orig_full, recon_full, title="", z_slice=None):
    """
    Muestra original FULL vs reconstrucción FULL (overlay).
    orig_full: (48,48,48) en intensidades originales
    recon_full: (48,48,48) en escala del modelo (típicamente [0,1])
    """
    if z_slice is None:
        z_slice = orig_full.shape[2] // 2

    plt.figure(title, figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(orig_full[:, :, z_slice], cmap="viridis")
    plt.title("Original FULL")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(recon_full[:, :, z_slice], cmap="viridis")
    plt.title("Recon FULL (overlay)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


vae.eval()
with torch.no_grad():
    # ---- CONTROL ----
    idx_ctrl = 0
    x_ctrl = x_tensor[idx_ctrl:idx_ctrl+1].to(device)  # ROI tensor (1,1,16,8,8)
    x_recon_c, mu_c, logvar_c, z_c = vae(x_ctrl)

    # a numpy ROI (16,8,8,1)
    y_roi_c = x_recon_c.cpu().numpy()                  # (1,1,16,8,8)
    y_roi_c = np.transpose(y_roi_c, (0,2,3,4,1))[0]    # (16,8,8,1)

# FULL original tal cual
orig_full_c = X[idx_ctrl].astype(np.float32)          # (48,48,48) ORIGINAL
# FULL reconstruido con overlay (fondo negro)
y_roi_c_denorm = denormalize_from_01(y_roi_c, norm_lo, norm_hi)  # escala original
recon_full_c = overlay_back_roi(y_roi_c_denorm, bbox, fill_value=0.0)

show_full_pair(orig_full_c, recon_full_c, title="FULL Control (48^3)")


with torch.no_grad():
    # ---- PARKINSON ----
    idx_pd = 45
    x_pd = x_tensor[idx_pd:idx_pd+1].to(device)
    x_recon_p, mu_p, logvar_p, z_p = vae(x_pd)

    y_roi_p = x_recon_p.cpu().numpy()
    y_roi_p = np.transpose(y_roi_p, (0,2,3,4,1))[0]    # (16,8,8,1)

orig_full_p = X[idx_pd].astype(np.float32)
y_roi_p_denorm = denormalize_from_01(y_roi_p, norm_lo, norm_hi)  # escala original
recon_full_p = overlay_back_roi(y_roi_p_denorm, bbox, fill_value=0.0)

show_full_pair(orig_full_p, recon_full_p, title="FULL Parkinson (48^3)")


# ==========================
# Diferencias (en ROI y FULL)
# ==========================

# Diferencia entre reconstrucciones FULL (ojo: incluye fondo negro)
dif_full_recon = np.mean(np.abs(recon_full_c - recon_full_p))
print("Dif media |Recon FULL CTRL - Recon FULL PD|:", dif_full_recon)

# Diferencia entre originales FULL tal cual
dif_full_input = np.mean(np.abs(orig_full_c - orig_full_p))
print("Dif media |Orig FULL CTRL - Orig FULL PD|:", dif_full_input)

# Error de reconstrucción SOLO dentro del ROI (más justo)
minimos, maximos, _ = bbox
roi_slice = (
    slice(minimos[0], maximos[0]+1),
    slice(minimos[1], maximos[1]+1),
    slice(minimos[2], maximos[2]+1),
)

orig_roi_c = orig_full_c[roi_slice]
orig_roi_p = orig_full_p[roi_slice]

recon_roi_c = recon_full_c[roi_slice]
recon_roi_p = recon_full_p[roi_slice]

err_roi_c = np.mean(np.abs(orig_roi_c - recon_roi_c))
err_roi_p = np.mean(np.abs(orig_roi_p - recon_roi_p))
print("Error recon ROI CTRL:", err_roi_c)
print("Error recon ROI PD  :", err_roi_p)

print(labels)



# ==========================
# Visualizar espacio latente
# ==========================

x_dummy = x_tensor.to(device)  # (N,1,48,48,48)
N = x_dummy.shape[0]
batch_size = 8   # o 8, 4... ajusta según tu GPU

mu_list = []
logvar_list = []
z_list = []

with torch.no_grad():
    for i in range(0, N, batch_size):
        x_batch = x_dummy[i:i+batch_size]
        mu_b, logvar_b, z_b = vae.encode(x_batch)
        mu_list.append(mu_b.cpu().numpy())
        logvar_list.append(logvar_b.cpu().numpy())
        z_list.append(z_b.cpu().numpy())

# Concatenamos todos los batches
mu_all = np.concatenate(mu_list, axis=0)        # (N, LATENT_DIM)
logvar_all = np.concatenate(logvar_list, axis=0)
z_all = np.concatenate(z_list, axis=0)


# -------------------------
# Plot espacio latente (μ)
# -------------------------

labels_num = (labels == "PD").astype(int)

# =========================================
# UMAP sobre μ (espacio latente del encoder)
# =========================================

# 1) (Opcional pero recomendado) Estandarizar las características
scaler_mu = StandardScaler()
mu_all_std = scaler_mu.fit_transform(mu_all)   # sigue siendo (N, LATENT_DIM)

# 2) Crear el modelo UMAP
reducer_mu = umap.UMAP(
    n_neighbors=15,      # estructura local vs global
    min_dist=0.1,        # cuán apretados quedan los clusters
    n_components=2,      # queremos 2D
    metric="euclidean",
    random_state=42
)

# 3) Ajustar y transformar
mu_umap = reducer_mu.fit_transform(mu_all_std)  # (N, 2)

# 4) Dibujar
plt.figure("UMAP del espacio latente (μ)")
plt.scatter(mu_umap[:, 0], mu_umap[:, 1],
            alpha=0.6, s=10, c=labels_num, cmap="coolwarm")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP 2D del espacio latente (μ)")
plt.grid(True)
plt.axis("equal")
plt.show()


# -------------------------
# Plot espacio latente (z)
# -------------------------

# =========================================
# UMAP sobre z (espacio latente muestreado)
# =========================================

scaler_z = StandardScaler()
z_all_std = scaler_z.fit_transform(z_all)

reducer_z = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
)

z_umap = reducer_z.fit_transform(z_all_std)  # (N, 2)

# ==========================
# Métricas de clustering
# ==========================

metrics = []

# Métricas en espacio latente (recomendado)
metrics.append(cluster_metrics(mu_all_std, labels_num, name="mu_std (latent)"))
metrics.append(cluster_metrics(z_all_std,  labels_num, name="z_std (latent)"))

# Métricas en UMAP 2D (proxy visual)
metrics.append(cluster_metrics(mu_umap, labels_num, name="mu_umap2D"))
metrics.append(cluster_metrics(z_umap,  labels_num, name="z_umap2D"))

df_metrics = pda.DataFrame(metrics)
print("\n===== MÉTRICAS CLUSTER =====")
print(df_metrics.to_string(index=False))



plt.figure("UMAP del espacio latente (z)")
plt.scatter(z_umap[:, 0], z_umap[:, 1],
            alpha=0.6, s=10, c=labels_num, cmap="coolwarm")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP 2D del espacio latente (z)")
plt.grid(True)
plt.axis("equal")
plt.show()

PESOS_LIST = [
    "mask_300Epochs_8BS_1e3LR_8LD_0_0001BStart_0_00005BEnd.pt", 
    "mask_300Epochs_8BS_1e3LR_8LD_0_001BStart_0_0005BEnd.pt", 
    "mask_300Epochs_8BS_1e3LR_4LD_0_0001BStart_0_00005BEnd.pt", 
    "mask_300Epochs_8BS_1e3LR_4LD_0_001BStart_0_0005BEnd.pt", 
    "mask_300Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt", 
    "mask_300Epochs_8BS_1e3LR_16LD_0_001BStart_0_0005BEnd.pt",
    "mask_600Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt" 
]
input_shape = (X_pad.shape[1], X_pad.shape[2], X_pad.shape[3])

all_rows = []

for idx, p in enumerate(PESOS_LIST, start=1):

    if not os.path.exists(p):
        print(f"[WARN] No existe: {p}")
        continue

    rows = eval_weights_cluster_metrics(
        pesos_path=p,
        x_tensor=x_tensor,
        labels_num=labels_num,
        device=device,
        input_shape=input_shape,
        batch_size=BATCH_SIZE
    )

    # Sobrescribimos el nombre por el índice
    for r in rows:
        r["name"] = str(idx)

    all_rows.extend(rows)

df_all = pda.DataFrame(all_rows)
print("\n===== COMPARATIVA ENTRENAMIENTOS (CLUSTER) =====")
print(df_all.to_string(index=False))
print(df_all.sort_values(["davies_bouldin", "silhouette"], ascending=[True, False]).to_string(index=False))
plot_cluster_metrics(df_all)

# ==========================
# GENERACIÓN: sampling en Z
# ==========================

num_samples = 6
z_full = X.shape[3] // 2  # ojo: X es (N,48,48,48) -> slice axial

with torch.no_grad():
    z_random = torch.randn(num_samples, LATENT_DIM).to(device)
    x_gen_roi = vae.decode(z_random)  # (num_samples,1,16,8,8)

    x_gen_roi_np = x_gen_roi.cpu().numpy()                    # (N,1,16,8,8)
    x_gen_roi_np = np.transpose(x_gen_roi_np, (0,2,3,4,1))    # (N,16,8,8,1)

# DENORMALIZAR A ESCALA ORIGINAL
x_gen_roi_denorm = denormalize_from_01(x_gen_roi_np, norm_lo, norm_hi)

# convertimos a FULL (fondo recomendado = norm_lo)
x_gen_full = np.stack([overlay_back_roi(x_gen_roi_denorm[i], bbox, fill_value=norm_lo)
                       for i in range(num_samples)], axis=0)

plt.figure("Muestras generadas (FULL 48^3)", figsize=(10,4))
for i in range(num_samples):
    plt.subplot(2, (num_samples+1)//2, i+1)
    plt.imshow(x_gen_full[i, :, :, z_full], cmap="viridis")
    plt.title(f"Sample {i}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ===================================================
# GENERACIÓN LOCAL: alrededor de un Control y un PD
# ===================================================

#Aqui vamos a hacer una prueba de pseudo-generacion, vamos a coger una muestra y
#pasarla por el encoder para obtener su version codificada, una vez tengamos eso
#vamos a añadirle variaciones a su espacio latente para ver si reconstruye cosas
#parecidas. Si observamos cosas identicas es porque el VAE esta colapsado y si 
#observamos cosa muy distintas es porque el vae no ha aprendido lo suficiente

num_variants = 5
z_full = X.shape[3] // 2

# ----- CONTROL -----
with torch.no_grad():
    idx_ctrl = 0
    x_ctrl = x_tensor[idx_ctrl:idx_ctrl+1].to(device)
    mu_c, logvar_c, _ = vae.encode(x_ctrl)

    sigma_c = torch.exp(0.5 * logvar_c)
    z_ctrl_variants = []
    for _ in range(num_variants):
        eps = torch.randn_like(mu_c)
        z_var = mu_c + 0.5 * sigma_c * eps
        z_ctrl_variants.append(z_var)
    z_ctrl_variants = torch.cat(z_ctrl_variants, dim=0)

    x_ctrl_variants_roi = vae.decode(z_ctrl_variants)  # (num_variants,1,16,8,8)
    x_ctrl_var_np = x_ctrl_variants_roi.cpu().numpy()  # en [0,1]
    x_ctrl_var_np = np.transpose(x_ctrl_var_np, (0,2,3,4,1))  # (num_variants,16,8,8,1)

# DENORMALIZAR variantes CTRL
x_ctrl_var_denorm = denormalize_from_01(x_ctrl_var_np, norm_lo, norm_hi)

# FULL original y FULL variantes
orig_full_ctrl = X[idx_ctrl].astype(np.float32)  # ORIGINAL tal cual
ctrl_full_variants = [overlay_back_roi(x_ctrl_var_denorm[i], bbox, fill_value=norm_lo)
                      for i in range(num_variants)]

plt.figure("Variantes alrededor de un Control (FULL)", figsize=(14,3))
plt.subplot(1, num_variants+1, 1)
plt.imshow(orig_full_ctrl[:, :, z_full], cmap="viridis")
plt.title("Original CTRL (FULL)")
plt.axis("off")

for i in range(num_variants):
    plt.subplot(1, num_variants+1, i+2)
    plt.imshow(ctrl_full_variants[i][:, :, z_full], cmap="viridis")
    plt.title(f"Var {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ----- PD -----
with torch.no_grad():
    idx_pd = 45
    x_pd = x_tensor[idx_pd:idx_pd+1].to(device)
    mu_p, logvar_p, _ = vae.encode(x_pd)

    sigma_p = torch.exp(0.5 * logvar_p)
    z_pd_variants = []
    for _ in range(num_variants):
        eps = torch.randn_like(mu_p)
        z_var = mu_p + 0.5 * sigma_p * eps
        z_pd_variants.append(z_var)
    z_pd_variants = torch.cat(z_pd_variants, dim=0)

    x_pd_variants_roi = vae.decode(z_pd_variants)
    x_pd_var_np = x_pd_variants_roi.cpu().numpy()        # en [0,1]
    x_pd_var_np = np.transpose(x_pd_var_np, (0,2,3,4,1)) # (num_variants,16,8,8,1)

# DENORMALIZAR variantes PD
x_pd_var_denorm = denormalize_from_01(x_pd_var_np, norm_lo, norm_hi)

orig_full_pd = X[idx_pd].astype(np.float32)
pd_full_variants = [overlay_back_roi(x_pd_var_denorm[i], bbox, fill_value=norm_lo)
                    for i in range(num_variants)]

plt.figure("Variantes alrededor de un PD (FULL)", figsize=(14,3))
plt.subplot(1, num_variants+1, 1)
plt.imshow(orig_full_pd[:, :, z_full], cmap="viridis")
plt.title("Original PD (FULL)")
plt.axis("off")

for i in range(num_variants):
    plt.subplot(1, num_variants+1, i+2)
    plt.imshow(pd_full_variants[i][:, :, z_full], cmap="viridis")
    plt.title(f"Var {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ================================
# Comprobaciones de colapso VAE
# ================================

z_var_per_dim = z_all.var(axis=0)
print("Varianza por dimensión latente:\n", z_var_per_dim)
print("Varianza media:", z_var_per_dim.mean())

from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(z_all))
print("Distancia media entre pares de z:", dist_matrix.mean())
print("Distancia mínima entre pares de z:", dist_matrix[dist_matrix>0].min())


# 2.3 Reconstrucciones de distintas entradas (mostrar FULL)
num_check = 5
idxs = np.linspace(0, len(x_tensor)-1, num_check, dtype=int)

with torch.no_grad():
    x_subset = x_tensor[idxs].to(device)
    _, _, z_subset = vae.encode(x_subset)
    x_recon_subset_roi = vae.decode(z_subset)  # (num_check,1,16,8,8)
    x_recon_subset_np = x_recon_subset_roi.cpu().numpy()               # en [0,1]
    x_recon_subset_np = np.transpose(x_recon_subset_np, (0,2,3,4,1))   # (num_check,16,8,8,1)

# DENORMALIZAR reconstrucciones
x_recon_subset_denorm = denormalize_from_01(x_recon_subset_np, norm_lo, norm_hi)

recon_full_subset = [overlay_back_roi(x_recon_subset_denorm[i], bbox, fill_value=norm_lo)
                     for i in range(num_check)]

plt.figure("Reconstrucciones FULL de distintas entradas", figsize=(14,5))
for i, idx in enumerate(idxs):
    plt.subplot(2, num_check, i+1)
    plt.imshow(X[idx, :, :, z_full], cmap="viridis")
    plt.title(f"Orig {idx}")
    plt.axis("off")

    plt.subplot(2, num_check, num_check + i + 1)
    plt.imshow(recon_full_subset[i][:, :, z_full], cmap="viridis")
    plt.title(f"Recon {idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================
# DISENTANGLING: variar una dimensión
# =====================================
"""
    Para una dimension dada, clona un numero de veces igual a n_steps el vector z
    y multiplica la sigma un numero de veces igual a [-3,3] de esta manera vemos
    en esa dimension lo que va cambiando, si la forma, las sombras...
"""
def plot_disentangling_for_dim_full(vae, x_tensor, X_full, bbox, idx_sample, dim, n_steps=7):
    vae.eval()
    z_full = X_full.shape[3] // 2

    with torch.no_grad():
        x = x_tensor[idx_sample:idx_sample+1].to(device)
        mu, logvar, _ = vae.encode(x)
        sigma = torch.exp(0.5 * logvar)

        factors = torch.linspace(-3.0, 3.0, n_steps).to(device)

        z_base = mu.repeat(n_steps, 1)
        z_var = z_base.clone()
        for i, f in enumerate(factors):
            z_var[i, dim] = mu[0, dim] + f * sigma[0, dim]

        x_var_roi = vae.decode(z_var)  # (n_steps,1,16,8,8)
        x_var_np = x_var_roi.cpu().numpy()               # en [0,1]
        x_var_np = np.transpose(x_var_np, (0,2,3,4,1))   # (n_steps,16,8,8,1)

    # DENORMALIZAR disentangling
    x_var_denorm = denormalize_from_01(x_var_np, norm_lo, norm_hi)

    # FULL (fondo recomendado = norm_lo)
    full_vars = [overlay_back_roi(x_var_denorm[i], bbox, fill_value=norm_lo) for i in range(n_steps)]
    orig_full = X_full[idx_sample].astype(np.float32)

    plt.figure(f"Disentangling FULL dim {dim} (sample {idx_sample})", figsize=(16,3))
    plt.subplot(1, n_steps+1, 1)
    plt.imshow(orig_full[:, :, z_full], cmap="viridis")
    plt.title("Original FULL")
    plt.axis("off")

    for i in range(n_steps):
        plt.subplot(1, n_steps+1, i+2)
        plt.imshow(full_vars[i][:, :, z_full], cmap="viridis")
        plt.title(f"{factors[i].item():.1f}σ")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


idx_ctrl = 0
for dim in range(min(4, LATENT_DIM)):
    plot_disentangling_for_dim_full(vae, x_tensor, X, bbox, idx_ctrl, dim, n_steps=7)

idx_pd = 45
for dim in range(min(4, LATENT_DIM)):
    plot_disentangling_for_dim_full(vae, x_tensor, X, bbox, idx_pd, dim, n_steps=7)


# =====================================
# CLASIFICACION SUPERVISADA BASADA EN COORDENADAS EN EL ESPACIO LATENTE
# =====================================


X_2d = z_umap          # o z_umap
y = labels_num
idx = 45

y_true, y_pred, y_proba, _ = predict_one_by_loo(X_2d, y, idx)
print("Etiqueta real:", "PD" if y_true==1 else "Control")
print("Predicción  :", "PD" if y_pred==1 else "Control")
if y_proba is not None:
    print("P(PD)     :", y_proba)

plot_one_prediction(X_2d, y, idx, y_true, y_pred, title="LOO sobre UMAP(μ)")
    
# =====================================
# SMV LINEAL + VALIDACION CRUZADA + METRICAS CLINICAS
# =====================================

X_features = z_all_std   # ← o z_all_std
y = labels_num

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

balanced_accs = []
sensitivities = []
specificities = []
aucs = []

mean_fpr = np.linspace(0, 1, 100)
tprs = []

for train_idx, test_idx in skf.split(X_features, y):

    X_train, X_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    balanced_accs.append(bal_acc)

    # Sensitivity = recall clase 1 (PD)
    sens = recall_score(y_test, y_pred, pos_label=1)
    sensitivities.append(sens)

    # Specificity = recall clase 0
    spec = recall_score(y_test, y_pred, pos_label=0)
    specificities.append(spec)

    # AUC
    auc = roc_auc_score(y_test, y_proba)
    aucs.append(auc)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

print("===== SVM LINEAR (5-Fold CV) =====")
print("Balanced Accuracy:", np.mean(balanced_accs))
print("Sensitivity (PD):", np.mean(sensitivities))
print("Specificity (Control):", np.mean(specificities))
print("AUC:", np.mean(aucs))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)

plt.figure(figsize=(6,6))
plt.plot(mean_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Linear SVM (CV)")
plt.legend()
plt.grid(True)
plt.show()















