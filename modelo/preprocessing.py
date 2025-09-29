# ===============================
# PREPROCESSING.PY
# ===============================

import os
import random
from shutil import copyfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
root_folder = '../PetImages'
cat_folder = os.path.join(root_folder, "Cat")
dog_folder = os.path.join(root_folder, "Dog")

base_dir = 'PetImages/cats_dogs'
for subdir in ['training/cats', 'training/dogs', 'validation/cats', 'validation/dogs', 'test/cats', 'test/dogs']:
    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

training_path = os.path.join(base_dir, 'training')
validation_path = os.path.join(base_dir, 'validation')
test_path = os.path.join(base_dir, 'test')


# Function to split data
def split_data(main_dir, training_dir, validation_dir, test_dir=None, include_test_split=True, split_size=0.9):
    files = [f for f in os.listdir(main_dir) if os.path.getsize(os.path.join(main_dir, f)) > 0]
    shuffled_files = random.sample(files, len(files))

    split = int(split_size * len(shuffled_files))
    train = shuffled_files[:split]
    split_valid_test = int(split + (len(shuffled_files) - split) / 2)

    if include_test_split:
        validation = shuffled_files[split:split_valid_test]
        test = shuffled_files[split_valid_test:]
    else:
        validation = shuffled_files[split:]
        test = []

    for f in train:
        copyfile(os.path.join(main_dir, f), os.path.join(training_dir, f))
    for f in validation:
        copyfile(os.path.join(main_dir, f), os.path.join(validation_dir, f))
    if include_test_split:
        for f in test:
            copyfile(os.path.join(main_dir, f), os.path.join(test_dir, f))

    print(f"Divisão: {main_dir}")


# Split cats and dogs
include_test = True
split_data(cat_folder, os.path.join(training_path, 'cats'), os.path.join(validation_path, 'cats'),
           os.path.join(test_path, 'cats'), include_test)
split_data(dog_folder, os.path.join(training_path, 'dogs'), os.path.join(validation_path, 'dogs'),
           os.path.join(test_path, 'dogs'), include_test)

# Transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = datasets.ImageFolder(root=training_path, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform) if include_test else None

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) if include_test else None

print("Processamento Concluido!")
