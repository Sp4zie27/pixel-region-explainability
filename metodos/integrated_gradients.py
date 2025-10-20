import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ===============================
# Pré-processamento
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
# Função de visualização
# ===============================
def visualize_integrated_gradients(attributions, image, title="Integrated Gradients"):
    attr = attributions.squeeze().cpu().detach().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    attr = np.max(np.abs(attr), axis=2)  # combina canais

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Imagem Original")

    plt.subplot(1,2,2)
    plt.imshow(attr, cmap='hot')
    plt.axis('off')
    plt.title(title)

    plt.show()

# ===============================
# Aplicando Integrated Gradients
# ===============================
image_path = "Imagens_teste/cats/54.jpg"
image_tensor, image = preprocess_image(image_path)

# Inicializa Integrated Gradients
ig = IntegratedGradients(model)

# Predição do modelo
model.eval()
output = model(image_tensor)
target_class = output.argmax(dim=1).item()

# Calcula atributos
attributions, delta = ig.attribute(image_tensor, target=target_class, return_convergence_delta=True)

# Visualiza
visualize_integrated_gradients(attributions, image, title=f"Integrated Gradients (Classe {target_class})")

# Mostra delta de convergência
print(f"Convergence delta: {delta.item():.4f}")
