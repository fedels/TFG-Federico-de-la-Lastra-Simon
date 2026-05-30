
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.svm import SVC
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    plt.title("Recon FULL")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

    

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
    
def plot_svm_linear_boundary_2d(X_2d, y, title="Frontera lineal SVM"):

    """
    Dibuja la frontera de decisión lineal de un SVM en un espacio 2D.

    X_2d: array de forma (N, 2)
    y: etiquetas binarias, 0 = Control, 1 = PD
    """

    clf = SVC(kernel="linear", probability=True, random_state=SEED)
    clf.fit(X_2d, y)

    # =========================
    # Crear malla 2D
    # =========================

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # =========================
    # Función de decisión
    # =========================

    decision_values = clf.decision_function(grid)
    decision_values = decision_values.reshape(xx.shape)

    # =========================
    # Figura
    # =========================

    plt.figure(figsize=(7, 6))

    # ==========================================
    # Frontera de decisión y márgenes SVM
    # ==========================================

    plt.contour(
        xx,
        yy,
        decision_values,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        linewidths=[1.5, 2.5, 1.5],
        colors="black"
    )

    # =========================
    # Controles -> azul
    # =========================

    plt.scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        c="blue",
        marker="o",
        s=35,
        alpha=0.8,
        label="Control"
    )

    # =========================
    # PD -> rojo
    # =========================

    plt.scatter(
        X_2d[y == 1, 0],
        X_2d[y == 1, 1],
        c="red",
        marker="o",
        s=35,
        alpha=0.8,
        label="PD"
    )

    # =========================
    # Opcional: support vectors
    # =========================
    """
    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Support vectors"
    )
    """

    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()

    return clf