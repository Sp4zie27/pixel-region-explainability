import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from numpy import trapezoid

# --------------------------- Configuração ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = "Imagens_teste/Imagens_avaliar"
steps = 10
grid_size = 10

# --------------------------- CNN ---------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Bloco 1
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Bloco 2
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Bloco 3 (Target layer para Grad-CAM)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Camadas densas
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu6(self.bn3(self.conv3(x)))
        x = F.relu6(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu6(self.bn5(self.conv5(x)))
        x = F.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu6(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --------------------------- Modelo ---------------------------

model_path = "../modelo/cnn_cats_dogs.pth"
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --------------------------- Pré-Processamento ---------------------------

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image


def plot_line_with_shadow(x, mean, std, label, color=None):
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

# --------------------------- Métodos de Explicabilidade ---------------------------

# Backpropagation
def backpropagation(model, image_tensor, target_class=None):
    image_tensor.requires_grad_()
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    loss.backward()
    saliency = image_tensor.grad.abs().squeeze().detach().cpu().numpy()
    saliency = saliency.max(axis=0)  # Pega canal mais importante
    return saliency, target_class

# Integrated Gradients
def integrated_gradients(model, image_tensor, target_class=None, baseline=None, steps=50):
    model.zero_grad()
    if baseline is None:
        baseline = torch.zeros_like(image_tensor).to(device)
    if target_class is None:
        target_class = model(image_tensor).argmax(dim=1).item()

    grads = []
    for i in range(steps + 1):
        alpha = float(i) / steps
        scaled = baseline + alpha * (image_tensor - baseline)
        scaled = scaled.clone().detach().requires_grad_(True)
        output = model(scaled)
        loss = output[0, target_class]
        loss.backward()
        grads.append(scaled.grad.detach().clone())
        model.zero_grad()

    grads = torch.stack(grads, dim=0)
    avg_grads = grads.mean(dim=0)
    ig = (image_tensor - baseline) * avg_grads
    ig_map = ig.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    ig_map = np.maximum(ig_map, 0)
    ig_map = ig_map.max(axis=2)
    return ig_map, target_class

# Smooth Grad
def smooth_grad(model, image_tensor, target_class=None, noise_levels=[0.1], num_samples=20):
    model.zero_grad()
    image_tensor = image_tensor.to(device)
    if target_class is None:
        target_class = model(image_tensor).argmax(dim=1).item()

    all_maps = []

    for noise in noise_levels:
        grad_sum = np.zeros_like(image_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy())
        for _ in range(num_samples):
            noisy_img = (image_tensor + noise * torch.randn_like(image_tensor).to(
                device)).clone().detach().requires_grad_(True)
            output = model(noisy_img)
            loss = output[0, target_class]
            loss.backward()
            grad = noisy_img.grad.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
            grad_sum += grad
            model.zero_grad()

        avg_grad = grad_sum / num_samples
        saliency = np.maximum(avg_grad, 0)
        saliency = saliency.max(axis=2)
        all_maps.append(saliency)

    return all_maps, noise_levels, target_class


def occlusion_map(model, image_tensor, patch_size=8, stride=4):
    _, _, H, W = image_tensor.shape
    model.eval()
    with torch.no_grad():
        base_output = torch.softmax(model(image_tensor), dim=1)
        base_conf, target_class = base_output.max(1)
        base_conf = base_conf.item()
        target_class = target_class.item()

    heatmap = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, i:i + patch_size, j:j + patch_size] = 0
            with torch.no_grad():
                conf = torch.softmax(model(occluded), dim=1)[0, target_class].item()
            drop = base_conf - conf
            heatmap[i:i + patch_size, j:j + patch_size] += drop

    heatmap /= heatmap.max() + 1e-8
    return heatmap, target_class

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam_map = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        grad_cam_map = F.interpolate(grad_cam_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map, class_idx

# --------------------------- Métricas de Avaliação ---------------------------

# Pixel Flipping
def pixel_flipping(model, image_tensor, backpropagation, target_class, steps=10):
    model.eval()
    image_tensor = image_tensor.clone().detach().to(device)
    saliency = backpropagation.copy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    H, W = saliency.shape
    flat_indices = np.argsort(-saliency.flatten())
    step_size = len(flat_indices) // steps

    confidences = [torch.softmax(model(image_tensor), dim=1)[0, target_class].item()]

    with torch.no_grad():
        for i in range(1, steps + 1):
            idx_to_zero = flat_indices[: i * step_size]
            perturbed = image_tensor.clone()
            perturbed_np = perturbed.cpu().numpy()
            for idx in idx_to_zero:
                y, x = divmod(idx, W)
                perturbed_np[0, :, y, x] = 0
            perturbed = torch.tensor(perturbed_np).to(device)
            conf = torch.softmax(model(perturbed), dim=1)[0, target_class].item()
            confidences.append(conf)

    return np.array(confidences[1:]), confidences[0]

# Region Perturbation
def region_perturbation(model, image_tensor, backpropagation, target_class, grid_size=10, visualize_every=10):
    model.eval()
    image_tensor = image_tensor.clone().detach().to(device)
    saliency = backpropagation.copy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    H, W = saliency.shape
    block_h = H // grid_size
    block_w = W // grid_size

    region_scores = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = saliency[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            score = block.mean()
            region_scores.append(((i, j), score))
    region_scores.sort(key=lambda x: -x[1])

    confidences = []
    with torch.no_grad():
        base_conf = torch.softmax(model(image_tensor), dim=1)[0, target_class].item()
        perturbed = image_tensor.clone()

        for idx in range(len(region_scores)):
            i, j = region_scores[idx][0]
            perturbed[:, :, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = 0
            conf = torch.softmax(model(perturbed), dim=1)[0, target_class].item()
            confidences.append(conf)

    reduced_confidences = []
    total_blocks = len(region_scores)

    for i in range(1, visualize_every + 1):
        index = int((i / visualize_every) * total_blocks) - 1
        reduced_confidences.append(confidences[index])

    return np.array(reduced_confidences), base_conf


# --------------------------- Procesasmento de Imagens ---------------------------

# Lista de imagens
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

# Dicionários para armazenar todas as confianças
pixel_flipping_all = {
    "Backpropagation": [],
    "Integrated Gradients": [],
    "SmoothGrad": [],
    "Occlusion Perturbation": [],
    "Grad-CAM": []
}

region_perturbation_all = {
    "Backpropagation": [],
    "Integrated Gradients": [],
    "SmoothGrad": [],
    "Occlusion Perturbation": [],
    "Grad-CAM": []
}

for image_path in tqdm(image_paths, desc="Processando imagens"):
    try:
        image_tensor, _ = preprocess_image(image_path)
    except Exception as e:
        print(f"Erro ao processar imagem {image_path}: {e}")
        continue

    # Determinar a classe prevista (target_class) apenas uma vez
    with torch.no_grad():
        output = model(image_tensor)
        target_class = output.argmax(dim=1).item()

# --------------------------- Geração dos Mapas de Importância ---------------------------

    saliency, _ = backpropagation(model, image_tensor, target_class=target_class)
    ig_map, _ = integrated_gradients(model, image_tensor, target_class=target_class, steps=50)
    smaps, _, _ = smooth_grad(model, image_tensor, target_class=target_class, noise_levels=[0.1], num_samples=20)
    smooth_map = smaps[0]
    occ_map, _ = occlusion_map(model, image_tensor, patch_size=8, stride=4)
    gradcam = GradCAM(model, model.conv6)  # Última conv layer
    gcam_map, _ = gradcam(image_tensor)

# --------------------------- Pixel Flipping ---------------------------

    pf_bp, _ = pixel_flipping(model, image_tensor, saliency, target_class, steps=steps)
    pf_ig, _ = pixel_flipping(model, image_tensor, ig_map, target_class, steps=steps)
    pf_sg, _ = pixel_flipping(model, image_tensor, smooth_map, target_class, steps=steps)
    pf_occ, _ = pixel_flipping(model, image_tensor, occ_map, target_class, steps=steps)
    pf_gcam, _ = pixel_flipping(model, image_tensor, gcam_map, target_class, steps=steps)

    pixel_flipping_all["Backpropagation"].append(pf_bp)
    pixel_flipping_all["Integrated Gradients"].append(pf_ig)
    pixel_flipping_all["SmoothGrad"].append(pf_sg)
    pixel_flipping_all["Occlusion Perturbation"].append(pf_occ)
    pixel_flipping_all["Grad-CAM"].append(pf_gcam)

# --------------------------- Region Perturbation ---------------------------

    rp_bp, _ = region_perturbation(model, image_tensor, saliency, target_class, grid_size=grid_size,visualize_every=steps)
    rp_ig, _ = region_perturbation(model, image_tensor, ig_map, target_class, grid_size=grid_size,visualize_every=steps)
    rp_sg, _ = region_perturbation(model, image_tensor, smooth_map, target_class, grid_size=grid_size,visualize_every=steps)
    rp_occ, _ = region_perturbation(model, image_tensor, occ_map, target_class, grid_size=grid_size,visualize_every=steps)
    rp_gcam, _ = region_perturbation(model, image_tensor, gcam_map, target_class, grid_size=grid_size,visualize_every=steps)

    region_perturbation_all["Backpropagation"].append(rp_bp)
    region_perturbation_all["Integrated Gradients"].append(rp_ig)
    region_perturbation_all["SmoothGrad"].append(rp_sg)
    region_perturbation_all["Occlusion Perturbation"].append(rp_occ)
    region_perturbation_all["Grad-CAM"].append(rp_gcam)

# --------------------------- Score Métodos/Métricas ---------------------------

x = np.linspace(10, 100, steps)

auc_scores_pf = {}
auc_scores_rp = {}
methods = list(pixel_flipping_all.keys())

# Score: Pixel Flipping
for method in methods:
    data = np.array(pixel_flipping_all[method])
    mean_confidence = data.mean(axis=0)
    score_auc = np.trapezoid(mean_confidence, x)
    auc_scores_pf[method] = score_auc

# Score: Region Perturbation
for method in methods:
    data = np.array(region_perturbation_all[method])
    mean_confidence = data.mean(axis=0)
    score_auc = np.trapezoid(mean_confidence, x)
    auc_scores_rp[method] = score_auc

df_scores = pd.DataFrame({
    "Pixel Flipping (AUC)": auc_scores_pf,
    "Region Perturbation (AUC)": auc_scores_rp
})

df_scores_sorted = df_scores.sort_values(by="Pixel Flipping (AUC)", ascending=True)

try:
    print(df_scores_sorted.to_markdown(floatfmt=".2f"))
except ImportError:
    print("AVISO: 'tabulate' não instalada. Imprimindo formato padrão.")
    print(df_scores_sorted)

# --------------------------- Gráficos ---------------------------

colors = ["blue", "green", "orange", "red", "purple"]

# Pixel Flipping
plt.figure(figsize=(18, 6))
for idx, method in enumerate(pixel_flipping_all.keys()):
    data = np.array(pixel_flipping_all[method])
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    plot_line_with_shadow(x, mean, std, label=method, color=colors[idx])
plt.xlabel("Progresso da Perturbação (%)", fontweight='bold')
plt.ylabel("Confiança", fontweight='bold')
plt.title("Pixel Flipping", fontweight='bold')
plt.legend(prop={'weight': 'bold'})
plt.xticks(np.arange(0, 101, 10), fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.show()

# Region Perturbation
plt.figure(figsize=(18, 6))
for idx, method in enumerate(region_perturbation_all.keys()):
    data = np.array(region_perturbation_all[method])
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    plot_line_with_shadow(x, mean, std, label=method, color=colors[idx])
plt.xlabel("Progresso da Perturbação (%)", fontweight='bold')
plt.ylabel("Confiança", fontweight='bold')
plt.title("Region Perturbation", fontweight='bold')
plt.legend(prop={'weight': 'bold'})
plt.xticks(np.arange(0, 101, 10), fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.show()
