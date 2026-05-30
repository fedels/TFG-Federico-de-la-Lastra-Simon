#==============================================================================
#
#***************************   LIBRERIAS   ************************************
#
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

from skimage import measure




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

EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LATENT_DIM = 4
BETA_START = 0.001
BETA_END = 0.0005

#==============================================================================
#
#*****************     REPOSITORIO DE PESOS GUARDADOS     *********************
#
#==============================================================================

#1
#PESOS = "etiqueta_300Epochs_8BS_1e3LR_8LD_0_0001BStart_0_00005BEnd.pt" 

#2
#PESOS = "etiqueta_300Epochs_8BS_1e3LR_8LD_0_001BStart_0_0005BEnd.pt"

#3
PESOS = "etiqueta_300Epochs_8BS_1e3LR_4LD_0_0001BStart_0_00005BEnd.pt"

#4
#PESOS = "etiqueta_300Epochs_8BS_1e3LR_4LD_0_001BStart_0_0005BEnd.pt"

#5
#PESOS = "etiqueta_300Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt"

#6
#PESOS = "etiqueta_300Epochs_8BS_1e3LR_16LD_0_001BStart_0_0005BEnd.pt"
#==============================================================================


#==============================================================================
#
#***************************   FUNCIONES   ************************************
#
#==============================================================================
def enmascarar(imagen, percentage=0.6):
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
    
    return Imask, v

def normalize_clip01(x, p_low =1, p_high=99):
    lo,hi = np.percentile(x,(p_low,p_high))
    x = np.clip(x,lo,hi)
    x = (x-lo)/(hi-lo+1e-8)
    print(lo)
    return x.astype(np.float32)



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


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps




class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)  
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)  
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        return x 


class Decoder3D(nn.Module):

    def __init__(self, latent_dim, feat_shape):

        super().__init__()
        self.latent_dim = latent_dim
        self.feat_shape = feat_shape  # (C, D, H, W)
        units = int(np.prod(feat_shape))  
        self.proj = nn.Linear(latent_dim + 1, units)
       


        self.dec_up1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1) 
        self.dec_up2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2,
                                          padding=1, output_padding=1) 
        self.dec_up3 = nn.ConvTranspose3d(16, 8,  kernel_size=3, stride=2,
                                          padding=1, output_padding=1)  
        self.output = nn.Conv3d(8, 1, kernel_size=1, padding=0)        

    def forward(self, z, y):
        # y: (N,) int64 -> (N,1) float
        y = y.view(-1, 1).float()
        z = torch.cat([z, y], dim=1)  # (N, latent_dim+1)

        N = z.size(0)
        x = F.relu(self.proj(z))
        x = x.view(N, *self.feat_shape)
        x = F.relu(self.dec_up1(x))
        x = F.relu(self.dec_up2(x))
        x = F.relu(self.dec_up3(x))
        x = torch.sigmoid(self.output(x))
        return x




class VAE(nn.Module):
    def __init__(self, latent_dim, beta = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.encoder_net = Encoder3D()
        dummy = torch.zeros(1, 2, 16, 8, 8)
        with torch.no_grad():
            feats = self.encoder_net(dummy)
        self.feat_shape = feats.shape[1:]           
        flat_dim = int(np.prod(self.feat_shape))    


        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.sampling = Sampling()
        self.decoder_net = Decoder3D(latent_dim, self.feat_shape)
        self.num_classes = 2  # Control vs PD
        self.classifier = nn.Linear(latent_dim, self.num_classes)

        # Peso del término de clasificación 
        self.lambda_cls = 0.5

    def encode(self, x, y):
        # Convertir y (N,) en volumen (N,1,D,H,W)
        N, _, D, H, W = x.shape
        y_vol = y.view(N,1,1,1,1).float().expand(N,1,D,H,W)

        # Concatenar como canal adicional
        x_cond = torch.cat([x, y_vol], dim=1)
        
        encoder_feats = self.encoder_net(x_cond)
        flat = encoder_feats.reshape(x.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.sampling(mu, logvar)
        return mu, logvar, z

    def decode(self, z, y):
        return self.decoder_net(z, y)

    def forward(self, x, y):
        mu, logvar, z = self.encode(x, y)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar, z


    def compute_loss(self, x, y):
        x_recon, mu, logvar, z = self.forward(x, y)

        # 1) Recon (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # 2) KL
        kl_div_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
       )
        kl_div_loss = torch.mean(kl_div_loss)

        # 3) Clasificación 
        logits = self.classifier(z)                 # (N, 2)
        cls_loss = F.cross_entropy(logits, y)        # y: (N,) int64

        total_loss = recon_loss + self.beta * kl_div_loss + self.lambda_cls * cls_loss
        return total_loss, recon_loss, kl_div_loss, cls_loss


    def fit(self, dataloader, optimizer, beta_sched,
            epochs, pesos_path, patience, dataset_size, device):

        best_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            beta_sched.on_epoch_begin(epoch)
            running_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for (batch_x, batch_y) in progress_bar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                total_loss, recon_loss, kl_div_loss, cls_loss = self.compute_loss(batch_x, batch_y)

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item() * batch_x.size(0)

                # Mostrar info en la barra
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_div_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "beta": f"{self.beta:.6f}",
                    })

            epoch_loss = running_loss / dataset_size
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}  "
                  f"(recon: {recon_loss.item():.6f}  "
                  f"KL: {kl_div_loss.item():.6f}  "
                  f"CLS: {cls_loss.item():.6f})")

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


#==============================================================================
#
#****************************     MAIN     ************************************
#
#==============================================================================

X_enm, _ = enmascarar(X)

print("Forma enm: ", X_enm.shape)

x_norm = normalize_clip01(X_enm)

X_pad, crop = pad_to_multiples(x_norm)
#En esta linea le añadimos el canal
X_pad = X_pad[..., np.newaxis].astype(np.float32)  # (N, D, H, W, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Creamos el VAE 
vae = VAE(LATENT_DIM).to(device)

# Datos a tensores PyTorch: (N, D,H,W,1) -> (N,1,D,H,W)
x_dummy_np = X_pad  
x_tensor = torch.from_numpy(np.transpose(x_dummy_np, (0,4,1,2,3)))  
# ---- AÑADIR: labels -> y_tensor ----
# PD=1, Control=0 (si tus labels son exactamente "PD" y "Control")
y_np = (labels == "PD").astype(np.int64)   # PD=1, resto=0
y_tensor = torch.from_numpy(y_np)          # (N,)

dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("DEBUG dataset sample lens:", len(dataset[0]))
print("DEBUG batch shapes:", next(iter(dataloader))[0].shape, next(iter(dataloader))[1].shape)

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

vae.eval()
with torch.no_grad():
    # Volumen correspondiente a un paciente Control 
    idx_ctrl = 0
    x_ctrl = x_tensor[idx_ctrl:idx_ctrl+1].to(device)
    y_ctrl = y_tensor[idx_ctrl:idx_ctrl+1].to(device)

    x_recon_c, mu_c, logvar_c, z_c = vae(x_ctrl, y_ctrl)
    y_dummy_c = x_recon_c.cpu().numpy()                 
    y_dummy_c = np.transpose(y_dummy_c, (0,2,3,4,1))    

z_slice = x_dummy_np.shape[2] // 2  

plt.figure("Figura Control")
plt.subplot(1,2,1); plt.imshow(x_dummy_np[0,:,:,z_slice,0], cmap="viridis"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(y_dummy_c[0,:,:,z_slice,0], cmap="viridis"); plt.title("Reconstruida VAE"); plt.axis("off")
plt.show()

with torch.no_grad():
    # Volumen correspondiente a un paciente Parkinson 
    idx_pd = 45
    x_pd = x_tensor[idx_pd:idx_pd+1].to(device)
    y_pd = y_tensor[idx_pd:idx_pd+1].to(device)

    x_recon_p, mu_p, logvar_p, z_p = vae(x_pd, y_pd)
    y_dummy_p = x_recon_p.cpu().numpy()
    y_dummy_p = np.transpose(y_dummy_p, (0,2,3,4,1))

plt.figure("Figura Parkinson")
plt.subplot(1,2,1); plt.imshow(x_dummy_np[45,:,:,z_slice,0], cmap="viridis"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(y_dummy_p[0,:,:,z_slice,0], cmap="viridis"); plt.title("Reconstruida VAE"); plt.axis("off")
plt.show()

"Vamos a comprobar la diferencia entre el volumen reconstruido control y parkinson para ver como de diferentes son"

dif = np.mean(np.abs(y_dummy_c-y_dummy_p))
print("La diferencia entre volumenes reconstruidos es: ", dif)

dif_input = np.mean(np.abs(x_dummy_np[0] - x_dummy_np[2]))
print("Diferencia entre entradas original Control y PD:", dif_input)

dif_ctrl = np.mean(np.abs(x_dummy_np[0] - y_dummy_c[0]))
dif_pd   = np.mean(np.abs(x_dummy_np[45] - y_dummy_p[0]))
print("Error recon Control:", dif_ctrl)
print("Error recon PD:", dif_pd)

"En esta lista de labels podemos ver qué valores son control y cuales son parkinson"
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
        y_batch = y_tensor[i:i+batch_size].to(device)

        mu_b, logvar_b, z_b = vae.encode(x_batch, y_batch)

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

# 1) Estandarizar las características
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

plt.figure("UMAP del espacio latente (z)")
plt.scatter(z_umap[:, 0], z_umap[:, 1],
            alpha=0.6, s=10, c=labels_num, cmap="coolwarm")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP 2D del espacio latente (z)")
plt.grid(True)
plt.axis("equal")
plt.show()


# ==========================
# GENERACIÓN: sampling en Z
# ==========================

num_samples = 6   # imágenes a generar
z_slice = X_pad.shape[2] // 2  # mismo corte axial que antes

with torch.no_grad():
    z_random = torch.randn(num_samples, LATENT_DIM).to(device)

    half = num_samples // 2
    y_gen = torch.cat([
        torch.zeros(half, dtype=torch.long),
        torch.ones(num_samples - half, dtype=torch.long)
    ], dim=0).to(device)

    x_gen = vae.decode(z_random, y_gen)      # tensor en GPU
    x_gen_np = x_gen.detach().cpu().numpy()  # numpy en CPU
    x_gen_np = np.transpose(x_gen_np, (0, 2, 3, 4, 1))

plt.figure("Muestras generadas desde N(0,I)")
for i in range(num_samples):
    plt.subplot(2, (num_samples+1)//2, i+1)
    plt.imshow(x_gen_np[i, :, :, z_slice, 0], cmap="viridis")
    plt.title(f"Sample {i} | y={int(y_gen[i].item())}")
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

num_variants = 5   # variaciones muestra

with torch.no_grad():
    # ----- CONTROL -----
    idx_ctrl = 0
    x_ctrl = x_tensor[idx_ctrl:idx_ctrl+1].to(device)
    y_ctrl = y_tensor[idx_ctrl:idx_ctrl+1].to(device)

    mu_c, logvar_c, z_c = vae.encode(x_ctrl, y_ctrl)

    # Hacemos copias de mu y sumamos ruido pequeño
    sigma_c = torch.exp(0.5 * logvar_c)
    z_ctrl_variants = []
    for _ in range(num_variants):
        eps = torch.randn_like(mu_c)
        z_var = mu_c + 0.5 * sigma_c * eps   # 0.5 controla cuánta variación
        z_ctrl_variants.append(z_var)
    z_ctrl_variants = torch.cat(z_ctrl_variants, dim=0)  # (num_variants, LATENT_DIM)

    y_ctrl_repeat = y_ctrl.repeat(num_variants)
    x_ctrl_variants = vae.decode(z_ctrl_variants, y_ctrl_repeat)
    x_ctrl_var_np = x_ctrl_variants.cpu().numpy()
    x_ctrl_var_np = np.transpose(x_ctrl_var_np, (0,2,3,4,1))

plt.figure("Variantes alrededor de un Control")
plt.subplot(1, num_variants+1, 1)
plt.imshow(X_pad[idx_ctrl,:,:,z_slice,0], cmap="viridis")
plt.title("Original CTRL")
plt.axis("off")

for i in range(num_variants):
    plt.subplot(1, num_variants+1, i+2)
    plt.imshow(x_ctrl_var_np[i,:,:,z_slice,0], cmap="viridis")
    plt.title(f"Var {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ----- PD -----
with torch.no_grad():
    # elige un índice PD real según tus labels
    idx_pd = 45
    x_pd = x_tensor[idx_pd:idx_pd+1].to(device)
    y_pd = y_tensor[idx_pd:idx_pd+1].to(device)

    mu_p, logvar_p, z_p = vae.encode(x_pd, y_pd)

    sigma_p = torch.exp(0.5 * logvar_p)
    z_pd_variants = []
    for _ in range(num_variants):
        eps = torch.randn_like(mu_p)
        z_var = mu_p + 0.5 * sigma_p * eps
        z_pd_variants.append(z_var)
    z_pd_variants = torch.cat(z_pd_variants, dim=0)

    y_pd_repeat = y_pd.repeat(num_variants)
    x_pd_variants = vae.decode(z_pd_variants, y_pd_repeat)
    x_pd_var_np = x_pd_variants.cpu().numpy()
    x_pd_var_np = np.transpose(x_pd_var_np, (0,2,3,4,1))

plt.figure("Variantes alrededor de un PD")
plt.subplot(1, num_variants+1, 1)
plt.imshow(X_pad[idx_pd,:,:,z_slice,0], cmap="viridis")
plt.title("Original PD")
plt.axis("off")

for i in range(num_variants):
    plt.subplot(1, num_variants+1, i+2)
    plt.imshow(x_pd_var_np[i,:,:,z_slice,0], cmap="viridis")
    plt.title(f"Var {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ================================
# Comprobaciones de colapso VAE
# ================================

# Varianza de z por dimensión
z_var_per_dim = z_all.var(axis=0)
print("Varianza por dimensión latente:\n", z_var_per_dim)
print("Varianza media:", z_var_per_dim.mean())

# Si muchas dimensiones tienen varianza cercana a 0 → no se usan realmente

# Distancias entre z de diferentes muestras
from scipy.spatial.distance import pdist, squareform

dist_matrix = squareform(pdist(z_all))  # matriz de distancias Euclídeas
print("Distancia media entre pares de z:", dist_matrix.mean())
print("Distancia mínima entre pares de z:", dist_matrix[dist_matrix>0].min())

# Generación desde varios z reales y comparar salidas
num_check = 5    # cuántas muestras comprobar
idxs = np.linspace(0, len(x_tensor)-1, num_check, dtype=int)

with torch.no_grad():
    x_subset = x_tensor[idxs].to(device)
    y_subset = y_tensor[idxs].to(device)

    _, _, z_subset = vae.encode(x_subset, y_subset)
    x_recon_subset = vae.decode(z_subset, y_subset)

    x_recon_subset_np = x_recon_subset.detach().cpu().numpy()
    x_recon_subset_np = np.transpose(x_recon_subset_np, (0,2,3,4,1))

plt.figure("Reconstrucciones de distintas entradas")
for i, idx in enumerate(idxs):
    plt.subplot(2, num_check, i+1)
    plt.imshow(X_pad[idx,:,:,z_slice,0], cmap="viridis")
    plt.title(f"Orig {idx}")
    plt.axis("off")

    plt.subplot(2, num_check, num_check + i + 1)
    plt.imshow(x_recon_subset_np[i,:,:,z_slice,0], cmap="viridis")
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
def plot_disentangling_for_dim(vae, x_tensor, y_tensor, idx_sample, dim, n_steps=7):
    """
    idx_sample: índice del volumen en x_tensor
    dim: índice de la dimensión latente que quieres variar (0,...,LATENT_DIM-1)
    n_steps: número de puntos entre -3σ y +3σ
    """
    vae.eval()
    z_slice = X_pad.shape[2] // 2

    with torch.no_grad():
        x = x_tensor[idx_sample:idx_sample+1].to(device)
        y = y_tensor[idx_sample:idx_sample+1].to(device)

        mu, logvar, _ = vae.encode(x, y)
        sigma = torch.exp(0.5 * logvar)

        factors = torch.linspace(-3.0, 3.0, n_steps).to(device)

        z_base = mu.repeat(n_steps, 1)
        z_var = z_base.clone()

        for i, f in enumerate(factors):
            z_var[i, dim] = mu[0, dim] + f * sigma[0, dim]

        # repetir y para n_steps decodes
        y_rep = y.repeat(n_steps)
        x_var = vae.decode(z_var, y_rep)

        x_var_np = x_var.detach().cpu().numpy()
        x_var_np = np.transpose(x_var_np, (0,2,3,4,1))

    plt.figure(f"Disentangling dim {dim} (sample {idx_sample})")
    plt.subplot(1, n_steps+1, 1)
    plt.imshow(X_pad[idx_sample,:,:,z_slice,0], cmap="viridis")
    plt.title("Original")
    plt.axis("off")

    for i in range(n_steps):
        plt.subplot(1, n_steps+1, i+2)
        plt.imshow(x_var_np[i,:,:,z_slice,0], cmap="viridis")
        plt.title(f"{factors[i].item():.1f}σ")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Ejemplo: probar en un Control (índice 0) variando varias dimensiones
idx_ctrl = 0
for dim in range(min(4, LATENT_DIM)):
    plot_disentangling_for_dim(vae, x_tensor, y_tensor, idx_ctrl, dim, n_steps=7)

idx_pd = 45
for dim in range(min(4, LATENT_DIM)):
    plot_disentangling_for_dim(vae, x_tensor, y_tensor, idx_pd, dim, n_steps=7)

