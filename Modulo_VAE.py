
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
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


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
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
        self.feat_shape = feat_shape  #Coge la forma del vector de salida del encoder (64, 2, 1, 1)
        units = int(np.prod(feat_shape)) #Calcula el numero de unidades que hay en total  64 * 2 * 1 * 1 = 128
        self.proj = nn.Linear(latent_dim, units) #Hace el vector lineal grande con el tamaño que debe tener para el primer paso de decodificacion
        # ConvTranspose3d para duplicar tamaño: usamos stride=2 y output_padding=1
        self.dec_up1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, #32 4 2 2
                                          padding=1, output_padding=1)  
        self.dec_up2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, #16 8 4 4
                                          padding=1, output_padding=1)  
        self.dec_up3 = nn.ConvTranspose3d(16, 8,  kernel_size=3, stride=2, #8 16 8 8
                                          padding=1, output_padding=1) 
        self.output = nn.Conv3d(8, 1, kernel_size=1, padding=0)         #1 16 8 8

    def forward(self, z):
        N = z.size(0)
        x = F.relu(self.proj(z))
        x = x.view(N, *self.feat_shape)  #Se recupera la forma del encoder
        x = F.relu(self.dec_up1(x)) #Comienza proceso de upsampling
        x = F.relu(self.dec_up2(x))
        x = F.relu(self.dec_up3(x))
        x = torch.sigmoid(self.output(x))
        return x  


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
        encoder_feats = self.encoder_net(x) 
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

        total_loss = recon_loss + self.beta * kl_div_loss
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