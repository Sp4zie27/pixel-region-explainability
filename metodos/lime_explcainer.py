import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

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
    return tensor, np.array(image.resize((150, 150)))


# ===============================
# 4️⃣ Função de previsão para o LIME
# ===============================
def batch_predict(images):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    batch = torch.stack([
        transform(Image.fromarray(img.astype("uint8"))).to(device)
        for img in images
    ])
    preds = model(batch)
    probs = F.softmax(preds, dim=1).detach().cpu().numpy()
    return probs


# ===============================
# 5️⃣ LIME
# ===============================
image_path = "Imagens_teste/cats/54.jpg"
image_tensor, image_np = preprocess_image(image_path)

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    image_np,
    batch_predict,
    top_labels=2,
    hide_color=0,
    num_samples=1000
)

target_class = explanation.top_labels[0]
lime_image_map, mask = explanation.get_image_and_mask(
    target_class,
    positive_only=True,
    hide_rest=False,
    num_features=8,  # número de segmentos a explicar
    min_weight=0.01
)

segments = np.unique(mask)
num_segments = len(segments)

print(f"Total de segmentos analisados: {num_segments}")


# ===============================
# 6️⃣ Função para confiança de um segmento
# ===============================
def confidence_for_segment(model, image_np, seg_mask, target_class):
    img_seg = image_np.copy()
    img_seg[seg_mask == 0] = 0
    tensor = torch.tensor(img_seg).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0, target_class].item()
    return prob


# ===============================
# 7️⃣ Visualização linha a linha + prints
# ===============================
fig, axes = plt.subplots(num_segments, 3, figsize=(12, 4 * num_segments))

if num_segments == 1:
    axes = np.expand_dims(axes, 0)  # garantir formato consistente

for i, seg_val in enumerate(segments):
    seg_mask = (mask == seg_val).astype(np.uint8)
    segmented_image = image_np.copy()
    segmented_image[seg_mask == 0] = 0

    # confiança do modelo apenas para este segmento
    conf_value = confidence_for_segment(model, image_np, seg_mask, target_class)

    # Print no console
    print(f"Segmento {i+1}/{num_segments} - Confiança: {conf_value:.4f}")

    # Coluna 1: Imagem original
    axes[i, 0].imshow(image_np)
    axes[i, 0].set_title("Imagem Original")
    axes[i, 0].axis("off")

    # Coluna 2: Segmento ativo
    axes[i, 1].imshow(segmented_image)
    axes[i, 1].set_title(f"Segmento {i+1}")
    axes[i, 1].axis("off")

    # Coluna 3: Gráfico de confiança
    axes[i, 2].bar(["Confiança"], [conf_value], color="tab:blue")
    axes[i, 2].set_ylim(0, 1)
    axes[i, 2].set_title(f"Confiança = {conf_value:.3f}")

plt.tight_layout()
plt.show()

