
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch
from sklearn.preprocessing import StandardScaler
import os
import Modulo_VAE as mod_vae


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
    mask[idx] = False #deja fuera la muestra deseada

    X_train, y_train = X_2d[mask], y[mask]
    X_test = X_2d[idx:idx+1]
    y_true = y[idx]

    model.fit(X_train, y_train)
    y_pred = int(model.predict(X_test)[0])

    # Probabilidad (si está disponible)
    y_proba = None
    if hasattr(model, "predict_proba"): #Se comprueba si el modelo seleccionado tiene dicha función
        y_proba = float(model.predict_proba(X_test)[0, 1])  # P(PD)

    return y_true, y_pred, y_proba, model


    
    

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
    vae, latent_dim_ckpt = mod_vae.build_vae_from_checkpoint(pesos_path, device, input_shape)

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

