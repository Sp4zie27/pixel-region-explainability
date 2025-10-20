import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Configuração do device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 1️⃣ Classe CNN
# ===============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

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

# ===============================
# 2️⃣ Configuração do device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 3️⃣ Classe CNN
# ===============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

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

# ===============================
# 4️⃣ Inicializar modelo e carregar pesos
# ===============================
model_path = "../modelo/cnn_cats_dogs.pth"
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===============================
# 5️⃣ Pré-processamento da imagem
# ===============================
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

# ===============================
# 6️⃣ Saliency Map
# ===============================
def saliency_map(model, image_tensor, target_class=None):
    image_tensor.requires_grad_()
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    loss.backward()
    saliency = image_tensor.grad.abs().squeeze().detach().cpu().numpy()
    saliency = saliency.max(axis=0)
    return saliency, target_class

# ===============================
# 7️⃣ Métricas: Pixel Flipping e Region Perturbation
# ===============================
def pixel_flipping(model, image_tensor, saliency, target_class, steps=10):
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    flat_saliency = saliency.flatten()
    sorted_idx = np.argsort(flat_saliency)[::-1]

    confidences = []
    total_pixels = len(sorted_idx)
    step_size = max(total_pixels // steps, 1)

    for i in range(0, total_pixels, step_size):
        perturbed = image_np.copy().reshape(-1, 3)
        perturbed[sorted_idx[:i]] = 0
        perturbed = perturbed.reshape(150, 150, 3)
        perturbed_tensor = torch.tensor(perturbed).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0, target_class].item()
        confidences.append(conf)

    return confidences

def region_perturbation(model, image_tensor, saliency, target_class, grid_size=10):
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    h, w, _ = image_np.shape
    region_h, region_w = h // grid_size, w // grid_size
    saliency_map_resized = saliency.reshape(h, w)

    region_importance = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region = saliency_map_resized[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            region_importance[i, j] = region.mean()

    sorted_regions = np.argsort(region_importance.flatten())[::-1]
    confidences = []
    perturbed = image_np.copy()

    for k in range(len(sorted_regions)):
        idx = sorted_regions[k]
        i, j = divmod(idx, grid_size)
        perturbed[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w, :] = 0
        perturbed_tensor = torch.tensor(perturbed).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0, target_class].item()
        confidences.append(conf)

    return confidences

# ===============================
# 8️⃣ Grad-CAM
# ===============================
def grad_cam(model, image_tensor, target_class=None, target_layer='conv6'):
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    image_tensor.requires_grad_()
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    loss = output[0, target_class]
    loss.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
    for i, w in enumerate(pooled_gradients):
        cam += w * activations[0, i, :, :]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = np.uint8(cam.cpu().numpy() * 255)
    cam = Image.fromarray(cam).resize(
        (image_tensor.shape[3], image_tensor.shape[2]), resample=Image.Resampling.LANCZOS
    )
    cam = np.array(cam) / 255.0
    return cam, target_class

# ===============================
# 9️⃣ Teste com imagem e visualizações
# ===============================
image_path = "Imagens_teste/dogs/5861.jpg"
image_tensor, image = preprocess_image(image_path)

# Saliency map
saliency, target_class = saliency_map(model, image_tensor)

# Classe prevista e confiança inicial
with torch.no_grad():
    probs = torch.softmax(model(image_tensor), dim=1).cpu().numpy()[0]
initial_conf = probs[target_class]
class_name = "Cão" if target_class == 1 else "Gato"
print(f"Imagem Prevista: {class_name}")
print(f"Confiança Inicial: {initial_conf:.4f}")

# Métricas
pixel_conf = pixel_flipping(model, image_tensor, saliency, target_class)
region_conf = region_perturbation(model, image_tensor, saliency, target_class)

print("\n--- Pixel Flipping ---")
print(f"Confiança inicial: {pixel_conf[0]:.4f}")
print(f"Confiança final após perturbação total: {pixel_conf[-1]:.4f}")
print(f"Queda de confiança: {pixel_conf[0]-pixel_conf[-1]:.4f}")

print("\n--- Region Perturbation ---")
print(f"Confiança inicial: {region_conf[0]:.4f}")
print(f"Confiança final após perturbação total: {region_conf[-1]:.4f}")
print(f"Queda de confiança: {region_conf[0]-region_conf[-1]:.4f}")

# Grad-CAM
cam, _ = grad_cam(model, image_tensor)

# Visualização
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Imagem Original ({class_name})")

plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title("Grad-CAM")


plt.subplot(1,3,3)
plt.plot(pixel_conf, label="Pixel Flipping")
plt.plot(region_conf, label="Region Perturbation")
plt.xlabel("Passos")
plt.ylabel("Confiança")
plt.legend()
plt.title("Avaliação Métricas")


plt.tight_layout()
plt.show()