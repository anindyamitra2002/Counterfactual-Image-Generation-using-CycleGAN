import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from models import ResUNetGenerator

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Function to save image
def save_image(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    array = tensor.permute(1, 2, 0).detach().numpy()
    array = (array * 0.5 + 0.5) * 255
    array = array.astype(np.uint8)
    if array.shape[2] == 1:
        array = array.squeeze(2)
        image = Image.fromarray(array, mode='L')
    else:
        image = Image.fromarray(array)
    image.save(path)

# Function to load model
def load_model(checkpoint_path, model_class, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_images(image_folder, g_NP_checkpoint, g_PN_checkpoint, output_dir='data/translated_images', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    g_NP = load_model(g_NP_checkpoint, lambda: ResUNetGenerator(gf=32, channels=1), device)
    g_PN = load_model(g_PN_checkpoint, lambda: ResUNetGenerator(gf=32, channels=1), device)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)

    # Collect image paths
    image_paths_0 = [os.path.join(image_folder, '0', fname) for fname in os.listdir(os.path.join(image_folder, '0')) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths_1 = [os.path.join(image_folder, '1', fname) for fname in os.listdir(os.path.join(image_folder, '1')) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
    dataset_0 = ImageDataset(image_paths_0, transform)
    dataset_1 = ImageDataset(image_paths_1, transform)
    dataloader_0 = DataLoader(dataset_0, batch_size=batch_size, shuffle=False)
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=False)
    
    # Process images from negative (0) to positive (1)
    with torch.no_grad():
        for batch, paths in tqdm(dataloader_0, desc="Converting N to P: "):
            batch = batch.to(device)
            translated_images = g_NP(batch)
            translated_images = g_PN(translated_images)
            for img, path in zip(translated_images, paths):
                save_path = os.path.join(output_dir, '1', os.path.basename(path))
                save_image(img, save_path)
        
        # Process images from positive (1) to negative (0)
        for batch, paths in tqdm(dataloader_1, desc="Converting P to N: "):
            batch = batch.to(device)
            translated_images = g_PN(batch)
            translated_images = g_NP(translated_images)
            for img, path in zip(translated_images, paths):
                save_path = os.path.join(output_dir, '0', os.path.basename(path))
                save_image(img, save_path)

if __name__ == '__main__':
    image_folder = r'data\rsna-pneumonia-dataset\train'
    g_NP_checkpoint = 'models\g_NP_best.ckpt'
    g_PN_checkpoint = 'models\g_PN_best.ckpt'
    
    
    generate_images(image_folder, g_NP_checkpoint, g_PN_checkpoint)
