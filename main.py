#==============================================================================
#
#***************************   LIBRERIAS   ************************************
#
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pda
import umap.umap_ as umap

#Modulos hechos por el autor del código
import Mask as mk
import Shaping as sh
import Modulo_VAE as mod_vae
import Metrics as mtk
import Plotting as pl


import torch
from torch.utils.data import TensorDataset, DataLoader


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

import random 

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#==============================================================================
#
#***************************     CONSTANTES     *******************************
#
#==============================================================================

#Nota: A mayor beta, mas castigo KL_loss u por tanto el modelo forzara a KL_loss 
#para que sea menor

EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LATENT_DIM = 4
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
PESOS = "mask_300Epochs_8BS_1e3LR_4LD_0_0001BStart_0_00005BEnd.pt" 

#4
#PESOS = "mask_300Epochs_8BS_1e3LR_4LD_0_001BStart_0_0005BEnd.pt" 

#5
#PESOS = "mask_300Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt" 

#6
#PESOS = "mask_300Epochs_8BS_1e3LR_16LD_0_001BStart_0_0005BEnd.pt" 

#7
#PESOS = "mask_600Epochs_8BS_1e3LR_16LD_0_0001BStart_0_00005BEnd.pt" 

#8
#PESOS = "prueba_borrar.pt" 

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
#****************************     MAIN     ************************************
#
#==============================================================================

X_enm, _, bbox = mk.enmascarar(X, return_bbox=True)
print("Forma X original:", X.shape)        # antes de enmascarar
print("Forma X_enm:", X_enm.shape)         # después de enmascarar (esto debería ser N,D,H,W o D,H,W)

x_norm, norm_lo, norm_hi = sh.normalize_clip01(X_enm, return_params=True)
print("Norm params: lo =", norm_lo, " hi =", norm_hi)
print("Forma x_norm:", x_norm.shape)       # mismo que X_enm

X_pad, crop = sh.pad_to_multiples(x_norm)
print("Forma X_pad (sin canal):", X_pad.shape)  # (N,D,H,W)

#En esta linea le añadimos el canal
X_pad = X_pad[..., np.newaxis].astype(np.float32)  # (N, D, H, W, 1)
print("Forma X_pad (con canal):", X_pad.shape)  # (N,D,H,W,1)

"X_pad es nuestos datos de entrada"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Creamos el VAE 
input_shape = (X_pad.shape[1], X_pad.shape[2], X_pad.shape[3]) 
vae = mod_vae.VAE(LATENT_DIM, input_shape=input_shape).to(device)

# Datos a tensores PyTorch: (N, D,H,W,1) -> (N,1,D,H,W)
x_dummy_np = X_pad  
x_tensor = torch.from_numpy(np.transpose(x_dummy_np, (0,4,1,2,3)))  
print("Forma x_tensor (PyTorch):", x_tensor.shape)  # (N,1,D,H,W)
dataset = TensorDataset(x_tensor)
g = torch.Generator()
g.manual_seed(SEED)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g
)

optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

beta_sched = mod_vae.BetaScheduler(vae,beta_start=BETA_START, beta_end=BETA_END,n_epochs=EPOCHS)

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
y_roi_c_denorm = sh.denormalize_from_01(y_roi_c, norm_lo, norm_hi)  # escala original
recon_full_c = sh.overlay_back_roi(y_roi_c_denorm, bbox, fill_value=0.0)

pl.show_full_pair(orig_full_c, recon_full_c, title="FULL Control (48^3)")


with torch.no_grad():
    # ---- PARKINSON ----
    idx_pd = 22
    x_pd = x_tensor[idx_pd:idx_pd+1].to(device)
    x_recon_p, mu_p, logvar_p, z_p = vae(x_pd)

    y_roi_p = x_recon_p.cpu().numpy()
    y_roi_p = np.transpose(y_roi_p, (0,2,3,4,1))[0]    # (16,8,8,1)

orig_full_p = X[idx_pd].astype(np.float32)
y_roi_p_denorm = sh.denormalize_from_01(y_roi_p, norm_lo, norm_hi)  # escala original
recon_full_p = sh.overlay_back_roi(y_roi_p_denorm, bbox, fill_value=0.0)

pl.show_full_pair(orig_full_p, recon_full_p, title="FULL Parkinson (48^3)")

# =========================================================
# ERROR MEDIO GLOBAL DE RECONSTRUCCIÓN POR CLASE:
#   1) SOLO EN LA ROI
#   2) EN EL VOLUMEN FULL COMPLETO
# =========================================================

vae.eval()

# Acumuladores globales
total_error_roi = 0.0
total_error_full = 0.0

# Acumuladores por clase
total_error_roi_ctrl = 0.0
total_error_roi_pd   = 0.0

total_error_full_ctrl = 0.0
total_error_full_pd   = 0.0

# Contadores por clase
num_ctrl = 0
num_pd   = 0

num_samples = x_tensor.shape[0]

# bbox -> para reconstruir FULL y extraer ROI
minimos, maximos, _ = bbox
roi_slice = (
    slice(minimos[0], maximos[0] + 1),
    slice(minimos[1], maximos[1] + 1),
    slice(minimos[2], maximos[2] + 1),
)

with torch.no_grad():
    for i in range(num_samples):

        # -------------------------------------------------
        # 1) Imagen original en ROI (normalizada, en x_tensor)
        # -------------------------------------------------
        x = x_tensor[i:i+1].to(device)               # (1,1,D,H,W)

        # -------------------------------------------------
        # 2) Reconstrucción ROI por el VAE
        # -------------------------------------------------
        x_recon, _, _, _ = vae(x)                    # (1,1,D,H,W)

        # -------------------------------------------------
        # 3) Pasar ambas a numpy y quitar batch/canal
        # -------------------------------------------------
        x_roi_np = x.cpu().numpy()[0, 0]             # (D,H,W)
        x_recon_roi_np = x_recon.cpu().numpy()[0, 0] # (D,H,W)

        # -------------------------------------------------
        # 4) Error medio absoluto SOLO en ROI
        # -------------------------------------------------
        err_roi = np.mean(np.abs(x_roi_np - x_recon_roi_np))
        total_error_roi += err_roi

        # -------------------------------------------------
        # 5) Pasar reconstrucción ROI a formato (D,H,W,1)
        # -------------------------------------------------
        x_recon_roi_np_4d = np.transpose(
            x_recon.cpu().numpy(), (0, 2, 3, 4, 1)
        )[0]                                         # (D,H,W,1)

        # -------------------------------------------------
        # 6) Desnormalizar la ROI reconstruida
        # -------------------------------------------------
        x_recon_roi_denorm = sh.denormalize_from_01(
            x_recon_roi_np_4d, norm_lo, norm_hi
        )

        # -------------------------------------------------
        # 7) Reconstruir volumen FULL con overlay
        # -------------------------------------------------
        recon_full = sh.overlay_back_roi(
            x_recon_roi_denorm, bbox, fill_value=0.0
        )

        # -------------------------------------------------
        # 8) Volumen FULL original
        # -------------------------------------------------
        orig_full = X[i].astype(np.float32)

        # -------------------------------------------------
        # 9) Error medio absoluto en FULL completo
        # -------------------------------------------------
        err_full = np.mean(np.abs(orig_full - recon_full))
        total_error_full += err_full

        # -------------------------------------------------
        # 10) Acumular por clase
        # -------------------------------------------------
        if labels[i] == "Control":
            total_error_roi_ctrl += err_roi
            total_error_full_ctrl += err_full
            num_ctrl += 1

        elif labels[i] == "PD":
            total_error_roi_pd += err_roi
            total_error_full_pd += err_full
            num_pd += 1

        # Si hay SWEDD u otras clases, no se acumulan aquí

# ---------------------------------------------------------
# 11) Promedios globales
# ---------------------------------------------------------
mean_error_roi = total_error_roi / num_samples
mean_error_full = total_error_full / num_samples

print("Error medio global de reconstrucción en ROI :", mean_error_roi)
print("Error medio global de reconstrucción en FULL:", mean_error_full)

# ---------------------------------------------------------
# 12) Promedios por clase
# ---------------------------------------------------------
if num_ctrl > 0:
    mean_error_roi_ctrl = total_error_roi_ctrl / num_ctrl
    mean_error_full_ctrl = total_error_full_ctrl / num_ctrl
    print("Error medio ROI   - Control:", mean_error_roi_ctrl)
    print("Error medio FULL  - Control:", mean_error_full_ctrl)
else:
    print("No hay muestras Control.")

if num_pd > 0:
    mean_error_roi_pd = total_error_roi_pd / num_pd
    mean_error_full_pd = total_error_full_pd / num_pd
    print("Error medio ROI   - PD:", mean_error_roi_pd)
    print("Error medio FULL  - PD:", mean_error_full_pd)
else:
    print("No hay muestras PD.")



# ==========================
# Visualizar espacio latente
# ==========================

x_dummy = x_tensor.to(device)  
N = x_dummy.shape[0]
batch_size = 8  

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

# 1) Estandarizar las características
scaler_mu = StandardScaler()
mu_all_std = scaler_mu.fit_transform(mu_all)   # sigue siendo (N, LATENT_DIM)

# 2) Crear el modelo UMAP
reducer_mu = umap.UMAP(
    n_neighbors=15,      # estructura local vs global
    min_dist=0.1,        # cuán apretados quedan los clusters
    n_components=2,      # queremos 2D
    metric="euclidean",
    random_state=SEED
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
    random_state=SEED
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
# Métricas de clustering
# ==========================

metrics = []

# Métricas en espacio latente 
metrics.append(mtk.cluster_metrics(mu_all_std, labels_num, name="mu_std (latent)"))
metrics.append(mtk.cluster_metrics(z_all_std,  labels_num, name="z_std (latent)"))

# Métricas en UMAP 2D (proxy visual)
metrics.append(mtk.cluster_metrics(mu_umap, labels_num, name="mu_umap2D"))
metrics.append(mtk.cluster_metrics(z_umap,  labels_num, name="z_umap2D"))

df_metrics = pda.DataFrame(metrics)
print("\n===== MÉTRICAS CLUSTER =====")
print(df_metrics.to_string(index=False))


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

    rows = mtk.eval_weights_cluster_metrics(
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
pl.plot_cluster_metrics(df_all)

# ==========================
# GENERACIÓN: sampling en Z
# ==========================

num_samples = 6
z_full = X.shape[3] // 2  

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
with torch.no_grad():
    z_random = torch.randn(num_samples, LATENT_DIM).to(device)
    x_gen_roi = vae.decode(z_random)  # (num_samples,1,16,8,8)

    x_gen_roi_np = x_gen_roi.cpu().numpy()                    # (N,1,16,8,8)
    x_gen_roi_np = np.transpose(x_gen_roi_np, (0,2,3,4,1))    # (N,16,8,8,1)

# DENORMALIZAR A ESCALA ORIGINAL
x_gen_roi_denorm = sh.denormalize_from_01(x_gen_roi_np, norm_lo, norm_hi)

# convertimos a FULL (fondo recomendado = norm_lo)
x_gen_full = np.stack([sh.overlay_back_roi(x_gen_roi_denorm[i], bbox, fill_value=norm_lo)
                       for i in range(num_samples)], axis=0)

plt.figure("Muestras generadas (FULL 48^3)", figsize=(10,4))
for i in range(num_samples):
    plt.subplot(2, (num_samples+1)//2, i+1)
    plt.imshow(x_gen_full[i, :, :, z_full], cmap="viridis")
    plt.title(f"Generación {i+1}")
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

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
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
x_ctrl_var_denorm = sh.denormalize_from_01(x_ctrl_var_np, norm_lo, norm_hi)

# FULL original y FULL variantes
orig_full_ctrl = X[idx_ctrl].astype(np.float32)  # ORIGINAL tal cual
ctrl_full_variants = [sh.overlay_back_roi(x_ctrl_var_denorm[i], bbox, fill_value=norm_lo)
                      for i in range(num_variants)]

plt.figure("Variantes alrededor de un Control (FULL)", figsize=(14,3))
plt.subplot(1, num_variants+1, 1)
plt.imshow(orig_full_ctrl[:, :, z_full], cmap="viridis")
plt.title("Original CTRL")
plt.axis("off")

for i in range(num_variants):
    plt.subplot(1, num_variants+1, i+2)
    plt.imshow(ctrl_full_variants[i][:, :, z_full], cmap="viridis")
    plt.title(f"Var {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# ----- PD -----
with torch.no_grad():
    idx_pd = 22
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
x_pd_var_denorm = sh.denormalize_from_01(x_pd_var_np, norm_lo, norm_hi)

orig_full_pd = X[idx_pd].astype(np.float32)
pd_full_variants = [sh.overlay_back_roi(x_pd_var_denorm[i], bbox, fill_value=norm_lo)
                    for i in range(num_variants)]

plt.figure("Variantes alrededor de un PD (FULL)", figsize=(14,3))
plt.subplot(1, num_variants+1, 1)
plt.imshow(orig_full_pd[:, :, z_full], cmap="viridis")
plt.title("Original PD")
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
#Si alguna varianza tiene valor cercano a 0, esa dimension apenas aporta informacion 
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
x_recon_subset_denorm = sh.denormalize_from_01(x_recon_subset_np, norm_lo, norm_hi)

recon_full_subset = [sh.overlay_back_roi(x_recon_subset_denorm[i], bbox, fill_value=norm_lo)
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
def plot_disentangling_for_dim_full(vae, x_tensor, X_full, bbox, idx_sample, dim, n_steps=4):
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
    x_var_denorm = sh.denormalize_from_01(x_var_np, norm_lo, norm_hi)

    # FULL (fondo recomendado = norm_lo)
    full_vars = [sh.overlay_back_roi(x_var_denorm[i], bbox, fill_value=norm_lo) for i in range(n_steps)]
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
    # =====================================================
    # MATRIZ DE ERRORES ENTRE TODAS LAS RECONSTRUCCIONES
    # =====================================================

    full_vars_np = np.stack(full_vars, axis=0)  # (n_steps, D, H, W)

    error_matrix = []

    for i in range(n_steps):
        row = []
        for j in range(n_steps):
            if i != j:
                diff = full_vars_np[i] - full_vars_np[j]
                mae = np.mean(np.abs(diff))  # puedes usar RMSE si prefieres
                row.append(mae)
        error_matrix.append(row)

    error_matrix = np.array(error_matrix)  # shape: (n_steps, n_steps-1)

    return error_matrix

matrices_ctrl = []
idx_ctrl = 0
for dim in range(min(4, LATENT_DIM)):
    M = plot_disentangling_for_dim_full(vae, x_tensor, X, bbox, idx_ctrl, dim, n_steps=4)
    matrices_ctrl.append(M)

for i in matrices_ctrl:
    print("Matriz control valor medio:",np.mean(i))

matrices_pd = []
idx_pd = 22
for dim in range(min(4, LATENT_DIM)):
    M = plot_disentangling_for_dim_full(vae, x_tensor, X, bbox, idx_pd, dim, n_steps=4)
    matrices_pd.append(M)

for i in matrices_pd:
    print("Matriz PD valor medio:",np.mean(i))

# =====================================
# CLASIFICACION SUPERVISADA BASADA EN COORDENADAS EN EL ESPACIO LATENTE
# =====================================


X_2d = z_umap          
y = labels_num
idx = 1429

correct_pd = 0
correct_ctrl = 0
correct = 0

for i in range(0,1429,1):
    y_true, y_pred, y_proba, _ = mtk.predict_one_by_loo(X_2d, y, i)
    if(y_true == 1):
        if(y_true == y_pred):
            correct_pd += 1
    if(y_true == 0):
        if(y_true == y_pred):
            correct_ctrl += 1
    if(y_true == y_pred):
        correct += 1
            
correct_mean_pd = correct_pd/np.sum(y == 1)
correct_mean_ctrl = correct_ctrl/np.sum(y == 0)
correct_mean = correct/np.size(y)

print("Numero de muestras pd: ", np.sum(y == 1))
print("correctos pd: ", correct_pd)
print("media de aciertos pd: ", correct_mean_pd)

print("Numero de muestras ctrl: ", np.sum(y == 0))
print("correctos ctrl: ", correct_ctrl)
print("media de aciertos ctrl: ", correct_mean_ctrl)

print("Numero de muestras : ", np.size(y))
print("correctos: ", correct)
print("media de aciertos : ", correct_mean)


y_true, y_pred, y_proba, _ = mtk.predict_one_by_loo(X_2d, y, idx)
print("Etiqueta real:", "PD" if y_true==1 else "Control")
print("Predicción  :", "PD" if y_pred==1 else "Control")
if y_proba is not None:
    print("Probabilidad de obtener PD: ", y_proba)

pl.plot_one_prediction(X_2d, y, idx, y_true, y_pred, title="LOO sobre UMAP(z)")
    
# =====================================
# SMV LINEAL + VALIDACION CRUZADA + METRICAS CLINICAS
# =====================================

X_features = z_all_std   # ← o z_all_std
y = labels_num

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

balanced_accs = []
sensitivities = []
specificities = []
aucs = []

mean_fpr = np.linspace(0, 1, 100)
tprs = []

for train_idx, test_idx in skf.split(X_features, y):

    X_train, X_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = SVC(kernel="linear", probability=True, random_state=SEED)
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

bal_acc_mean = np.mean(balanced_accs)
bal_acc_std  = np.std(balanced_accs, ddof=1)

sens_mean = np.mean(sensitivities)
sens_std  = np.std(sensitivities, ddof=1)

spec_mean = np.mean(specificities)
spec_std  = np.std(specificities, ddof=1)

auc_mean = np.mean(aucs)
auc_std  = np.std(aucs, ddof=1)

print("===== SVM LINEAR (5-Fold CV) =====")
print(f"Balanced Accuracy:      {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
print(f"Sensitivity (PD):       {sens_mean:.4f} ± {sens_std:.4f}")
print(f"Specificity (Control):  {spec_mean:.4f} ± {spec_std:.4f}")
print(f"AUC:                    {auc_mean:.4f} ± {auc_std:.4f}")

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)

plt.figure(figsize=(6,6))
plt.plot(mean_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# VISUALIZACIÓN DE LA FRONTERA LINEAL DEL SVM EN 2D
# =====================================================


X_svm_2d = z_umap   # también puedes probar con mu_umap

clf_visual = pl.plot_svm_linear_boundary_2d(
    X_svm_2d,
    labels_num,
    title="Frontera lineal SVM sobre UMAP(z)"
)



















