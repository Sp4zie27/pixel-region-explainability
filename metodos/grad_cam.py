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


# --------------------------- Grad-Cam ---------------------------


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

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        grad_cam_map = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        grad_cam_map = F.interpolate(grad_cam_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map, class_idx


# --------------------------- Pixel-Flipping ---------------------------


def pixel_flipping(model, image_tensor, grad_map, target_class, steps=100, visualize_every=10):
    image_np = image_tensor.squeeze().permute(1,2,0).cpu().numpy()
    flat_map = grad_map.flatten()
    sorted_idx = np.argsort(flat_map)[::-1]

    confidences = []
    perturbed_images = []
    total_pixels = len(flat_map)
    step_size = max(total_pixels // steps, 1)

    print("\n=== Pixel Flipping ===")
    for step, i in enumerate(range(0, total_pixels, step_size)):
        perturbed = image_np.copy().reshape(-1, 3)
        idxs = sorted_idx[:min(i, total_pixels)]
        perturbed[idxs] = 0
        perturbed = perturbed.reshape(150,150,3)

        perturbed_tensor = torch.tensor(perturbed).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0,target_class].item()
        confidences.append(conf)

        if step % visualize_every == 0:
            print(f"Passo {step:3d} → Confiança: {conf:.4f}")
            perturbed_images.append((perturbed.copy(), conf, step))

    return confidences, perturbed_images


# --------------------------- Region Pertubacion ---------------------------


def region_perturbation(model, image_tensor, grad_map, target_class, grid_size=10, visualize_every=10):
    image_np = image_tensor.squeeze().permute(1,2,0).cpu().numpy()
    h, w, _ = image_np.shape
    region_h, region_w = h // grid_size, w // grid_size

    grad_resized = grad_map
    region_importance = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region = grad_resized[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            region_importance[i,j] = region.mean()

    sorted_regions = np.argsort(region_importance.flatten())[::-1]
    confidences = []
    perturbed_images = []
    perturbed = image_np.copy()

    print("\n=== Region Perturbation ===")
    for step, idx in enumerate(sorted_regions):
        i,j = divmod(idx, grid_size)
        perturbed[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w, :] = 0
        perturbed_tensor = torch.tensor(perturbed).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            conf = torch.softmax(model(perturbed_tensor), dim=1)[0,target_class].item()
        confidences.append(conf)

        if step % visualize_every == 0:
            print(f"Região {step:3d} → Confiança: {conf:.4f}")
            perturbed_images.append((perturbed.copy(), conf, step))

    return confidences, perturbed_images


# --------------------------- Teste Imagem ---------------------------


image_path = "Imagens_teste/cats/1278.jpg"

image_tensor, image = preprocess_image(image_path)

grad_cam = GradCAM(model, model.conv6)
grad_map, target_class = grad_cam(image_tensor)

with torch.no_grad():
    probs = torch.softmax(model(image_tensor), dim=1).cpu().numpy()[0]
initial_conf = probs[target_class]
class_name = "Cão" if target_class==1 else "Gato"
print(f"\nImagem Prevista: {class_name}")
print(f"Confiança Inicial: {initial_conf:.4f}")

# Executar métricas
pixel_conf, pixel_imgs = pixel_flipping(model, image_tensor, grad_map, target_class, steps=100, visualize_every=10)
region_conf, region_imgs = region_perturbation(model, image_tensor, grad_map, target_class, grid_size=10, visualize_every=10)


# --------------------------- Visualizações Gráficas ---------------------------


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Imagem Original\nClasse: {class_name}   Conf: {initial_conf:.4f}")

from PIL import Image as PILImage
grad_map_img = PILImage.fromarray((grad_map * 255).astype(np.uint8))
grad_map_img = grad_map_img.resize(image.size, resample=PILImage.BILINEAR)
grad_map_resized = np.array(grad_map_img) / 255.0

plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(grad_map_resized, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title("Grad-CAM")

plt.subplot(1,3,3)
plt.plot(pixel_conf, label="Pixel Flipping")
plt.plot(region_conf, label="Region Perturbation")
plt.xlabel("Progresso da Perturbação (%)")
plt.ylabel("Confiança")
plt.legend()
plt.title("Evolução da Confiança")
plt.tight_layout()
plt.show()

def show_evolution(images_list, title):
    plt.figure(figsize=(15,8))
    num_imgs = len(images_list)
    for idx, (img, conf, step) in enumerate(images_list):
        plt.subplot(2, (num_imgs+1)//2, idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Remoção: {step}%\nConf: {conf:.2f}")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_evolution(pixel_imgs, "Evolução da Imagem - Pixel Flipping")
show_evolution(region_imgs, "Evolução da Imagem - Region Perturbation")