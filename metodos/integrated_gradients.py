import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- CNN ---------------------------


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


# --------------------------- Integrated Gradients ---------------------------


def integrated_gradients(model, image_tensor, target_class=None, baseline=None, steps=50):
    model.zero_grad()

    if baseline is None:
        baseline = torch.zeros_like(image_tensor).to(device)

    if target_class is None:
        target_class = model(image_tensor).argmax(dim=1).item()

    scaled_inputs = [baseline + (float(i) / steps) * (image_tensor - baseline) for i in range(steps + 1)]
    grads = []

    for scaled in scaled_inputs:
        scaled.requires_grad_()
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
    return ig_map, target_class


# --------------------------- Pixel Flipping ---------------------------


def pixel_flipping(model, image_tensor, importance_map, target_class, steps=100, visualize_every=10):
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    flat_map = importance_map.mean(axis=-1).flatten()
    sorted_idx = np.argsort(flat_map)[::-1]

    confidences = []
    images_to_show = []
    total_pixels = len(sorted_idx)
    step_size = max(total_pixels // steps, 1)

    print("\n=== Pixel Flipping ===")
    for step, i in enumerate(range(0, total_pixels, step_size), start=1):
        perturbed = image_np.copy().reshape(-1, 3)
        perturbed[sorted_idx[:i]] = 0
        perturbed = perturbed.reshape(150, 150, 3)
        perturbed_tensor = torch.tensor(perturbed).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0, target_class].item()
        confidences.append(conf)

        if step % visualize_every == 0:
            print(f"Passo {step} - Confiança: {conf:.4f}")
            images_to_show.append((perturbed.copy(), conf, step))

    return confidences, images_to_show


# --------------------------- Region Pertubacion ---------------------------


def region_perturbation(model, image_tensor, importance_map, target_class, grid_size=10, visualize_every=10):
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    h, w, _ = image_np.shape
    region_h, region_w = h // grid_size, w // grid_size
    resized_map = importance_map.mean(axis=-1).reshape(h, w)

    region_importance = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region = resized_map[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            region_importance[i, j] = region.mean()

    sorted_regions = np.argsort(region_importance.flatten())[::-1]
    confidences = []
    images_to_show = []
    perturbed = image_np.copy()

    print("\n=== Region Perturbation ===")
    for step, idx in enumerate(sorted_regions, start=1):
        i, j = divmod(idx, grid_size)
        perturbed[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w, :] = 0
        perturbed_tensor = torch.tensor(perturbed).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0, target_class].item()
        confidences.append(conf)

        if step % visualize_every == 0:
            print(f"Passo {step} - Confiança: {conf:.4f}")
            images_to_show.append((perturbed.copy(), conf, step))

    return confidences, images_to_show


# --------------------------- Teste Imagem ---------------------------

image_path = "Imagens_teste/cats/1278.jpg"
image_tensor, image = preprocess_image(image_path)
ig_map, target_class = integrated_gradients(model, image_tensor)

# Confiança inicial
with torch.no_grad():
    probs = torch.softmax(model(image_tensor), dim=1).cpu().numpy()[0]
initial_conf = probs[target_class]
class_name = "Cão" if target_class == 1 else "Gato"

print(f"\nImagem Prevista: {class_name}")
print(f"Confiança Inicial: {initial_conf:.4f}")

# Métricas
pixel_conf, pixel_imgs = pixel_flipping(model, image_tensor, ig_map, target_class)
region_conf, region_imgs = region_perturbation(model, image_tensor, ig_map, target_class)


# --------------------------- Visualização Gráfica ---------------------------

image_resized = np.array(image.resize((150, 150))) / 255.0
ig_map_norm = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min())
gradients_at_image = np.clip(image_resized * ig_map_norm * 3, 0, 1)

plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Imagem Original ({class_name})")

plt.subplot(1, 4, 2)
plt.imshow(ig_map_norm, cmap='hot')
plt.axis('off')
plt.title("Integrated Gradients")

plt.subplot(1, 4, 3)
plt.imshow(gradients_at_image)
plt.axis('off')
plt.title("Gradients at Image")

plt.subplot(1, 4, 4)
plt.plot(pixel_conf, label="Pixel Flipping")
plt.plot(region_conf, label="Region Perturbation")
plt.xlabel("Progresso da Perturbação (%)")
plt.ylabel("Confiança")
plt.legend()
plt.title("Avaliação de Métricas")

plt.tight_layout()
plt.show()

def show_evolution(images_list, title):
    plt.figure(figsize=(20, 8))
    for idx, (img, conf, step) in enumerate(images_list):
        plt.subplot(2, (len(images_list) + 1) // 2, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Passo {step}\nConf: {conf:.2f}")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_evolution(pixel_imgs, "Evolução da Imagem - Pixel Flipping")
show_evolution(region_imgs, "Evolução da Imagem - Region Perturbation")
