import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr

pixel_flipping_data = {
    "Backpropagation": [0.0782, 0.1429, 0.0905, 0.2234, 0.4037, 0.8794, 0.7598, 0.8986, 0.7927, 0.4273],
    "Integrated Gradients": [0.2799, 0.0581, 0.2663, 0.5477, 0.6633, 0.6850, 0.6867, 0.5085, 0.6791, 0.1695],
    "SmoothGrad": [0.0782, 0.1429, 0.0905, 0.2234, 0.4037, 0.8794, 0.7598, 0.8986, 0.7927, 0.4273],
    "Occlusion Perturbation": [0.9214, 0.5346, 0.1673, 0.2864, 0.1857, 0.4778, 0.2828, 0.3091, 0.3418, 0.2492],
    "Grad-CAM": [1.0000, 0.8502, 0.9214, 0.9793, 0.8741, 0.6515, 0.5808, 0.5776, 0.5204, 0.5776]
}

region_perturbation_data = {
    "Backpropagation": [0.8529, 0.7203, 0.7536, 0.7868, 0.6038, 0.5388, 0.3896, 0.3221, 0.2738, 0.2935],
    "Integrated Gradients": [0.8984, 0.7928, 0.7586, 0.7093, 0.5991, 0.3957, 0.3883, 0.3374, 0.3258, 0.2935],
    "SmoothGrad": [0.8529, 0.6943, 0.5971, 0.8418, 0.6605, 0.5388, 0.3896, 0.3221, 0.2738, 0.2935],
    "Occlusion Perturbation": [0.9201, 0.6729, 0.4178, 0.5416, 0.4845, 0.5073, 0.4694, 0.3556, 0.3093, 0.2935],
    "Grad-CAM": [1.0000, 0.8473, 0.8104, 0.3103, 0.2146, 0.4964, 0.3736, 0.2343, 0.2400, 0.2007]
}

steps = np.arange(10, 110, 10)


# --------------------------- HeatMaps Evolução ---------------------------


plt.figure(figsize=(20,6))
sns.heatmap(
    pd.DataFrame(pixel_flipping_data, index=steps).T,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 14, "weight": "bold"}  # 🔹 números grandes e negrito
)
plt.title("Heatmap (Pixel Flipping)", fontsize=18, fontweight="bold")
plt.xlabel("Remoção (%)", fontsize=16, fontweight="bold")
plt.ylabel("Métodos", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13, fontweight="bold")
plt.yticks(fontsize=13, fontweight="bold")
plt.show()

plt.figure(figsize=(20,6))
sns.heatmap(
    pd.DataFrame(region_perturbation_data, index=steps).T,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 14, "weight": "bold"}  # 🔹 números grandes e negrito
)
plt.title("Heatmap (Region Perturbation)", fontsize=18, fontweight="bold")
plt.xlabel("Remoção (%)", fontsize=16, fontweight="bold")
plt.ylabel("Métodos", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13, fontweight="bold")
plt.yticks(fontsize=13, fontweight="bold")
plt.show()


# --------------------------- HeatMaps Correlação ---------------------------


corr_pixel = pd.DataFrame(pixel_flipping_data).corr(method="spearman")
corr_region = pd.DataFrame(region_perturbation_data).corr(method="spearman")

plt.figure(figsize=(20,8))
sns.heatmap(
    corr_pixel,
    annot=True,
    cmap="coolwarm",
    annot_kws={"size": 16, "weight": "bold"}  # 🔹 números grandes e negrito
)
plt.title("Correlação entre métodos (Pixel Flipping)", fontsize=18, fontweight="bold")
plt.xlabel("Métodos", fontsize=16, fontweight="bold")
plt.ylabel("Métodos", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13, fontweight="bold")
plt.yticks(fontsize=13, fontweight="bold")
plt.show()

plt.figure(figsize=(20,8))
sns.heatmap(
    corr_region,
    annot=True,
    cmap="coolwarm",
    annot_kws={"size": 16, "weight": "bold"}  # 🔹 números grandes e negrito
)
plt.title("Correlação entre métodos (Region Perturbation)", fontsize=18, fontweight="bold")
plt.xlabel("Métodos", fontsize=16, fontweight="bold")
plt.ylabel("Métodos", fontsize=16, fontweight="bold")
plt.xticks(fontsize=13, fontweight="bold")
plt.yticks(fontsize=13, fontweight="bold")
plt.show()
