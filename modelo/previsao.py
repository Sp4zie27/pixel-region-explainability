import torch
from torch.serialization import safe_globals
from torchvision import transforms
from PIL import Image

class_names = ['Cat', 'Dog']

# ===============================
# Carregar modelo completo com safe_globals
# ===============================
with safe_globals():  # permite carregar globals confiáveis
    model = torch.load("cnn_cats_dogs_full.pth", weights_only=False)

model.eval()

# ===============================
# Transformações da imagem
# ===============================
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# ===============================
# Função de previsão
# ===============================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    prediction = class_names[predicted.item()]
    print(f"Prediction: {prediction}")
    return prediction

# ===============================
# Teste
# ===============================
if __name__ == "__main__":
    img_path = "../PetImages/Cat/1.jpg"
    predict_image(img_path)
