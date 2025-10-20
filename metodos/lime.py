import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries

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
# 2️⃣ Inicializar modelo
# ===============================
model_path = "../modelo/cnn_cats_dogs.pth"
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===============================
# 3️⃣ Pré-processamento da imagem
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
# 4️⃣ LIME Explainer
# ===============================
def lime_explanation(model, image, target_class=None, num_samples=1000, segmentation_fn=None):
    # Função de predição para LIME (entrada numpy HWC 0-255)
    def predict(images):
        images = np.array(images).astype(np.float32) / 255.0
        images_tensor = torch.tensor(images).permute(0,3,1,2).float().to(device)
        with torch.no_grad():
            probs = torch.softmax(model(images_tensor), dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(image),
        classifier_fn=predict,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn
    )

    if target_class is None:
        target_class = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        label=target_class,
        positive_only=True,
        hide_rest=False,
        num_features=1000
    )

    saliency = mask.astype(float)
    saliency = saliency / saliency.max()  # normaliza para 0-1
    return saliency, target_class

# ===============================
# 5️⃣ Métricas
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
# 6️⃣ Teste com imagem
# ===============================
image_path = "Imagens_teste/cats/54.jpg"
image_tensor, image = preprocess_image(image_path)
saliency, target_class = lime_explanation(model, image)

# Classe prevista e confiança inicial
with torch.no_grad():
    probs = torch.softmax(model(image_tensor), dim=1).cpu().numpy()[0]
initial_conf = probs[target_class]
class_name = "Cão" if target_class == 1 else "Gato"
print(f"Imagem Prevista: {class_name}")
print(f"Confiança Inicial: {initial_conf:.4f}")

# Pixel Flipping
pixel_conf = pixel_flipping(model, image_tensor, saliency, target_class)
print("\n--- Pixel Flipping ---")
print(f"Confiança inicial: {pixel_conf[0]:.4f}")
print(f"Confiança final após perturbação total: {pixel_conf[-1]:.4f}")
print(f"Queda de confiança: {pixel_conf[0]-pixel_conf[-1]:.4f}")

# Region Perturbation
region_conf = region_perturbation(model, image_tensor, saliency, target_class)
print("\n--- Region Perturbation ---")
print(f"Confiança inicial: {region_conf[0]:.4f}")
print(f"Confiança final após perturbação total: {region_conf[-1]:.4f}")
print(f"Queda de confiança: {region_conf[0]-region_conf[-1]:.4f}")

# ===============================
# 7️⃣ Visualização
# ===============================
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Imagem Original ({class_name})")

plt.subplot(1,3,2)
plt.imshow(saliency, cmap='hot')
plt.axis('off')
plt.title("LIME Map")

plt.subplot(1,3,3)
plt.plot(pixel_conf, label="Pixel Flipping")
plt.plot(region_conf, label="Region Perturbation")
plt.xlabel("Passos")
plt.ylabel("Confiança")
plt.legend()
plt.title("Avaliação Métricas")

plt.tight_layout()
plt.show()
