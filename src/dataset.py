from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os 

class ClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ['0', '1']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, label
    


class CustomDataset(Dataset):
    def __init__(self, root_dir, train_N, train_P, img_res):
        self.root_dir = root_dir
        self.train_N = train_N
        self.train_P = train_P
        self.img_res = img_res
        self.transforms = transforms.Compose([
            transforms.Resize(img_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
        ])

    def __len__(self):
        return min(len(os.listdir(os.path.join(self.root_dir, self.train_N))),
                   len(os.listdir(os.path.join(self.root_dir, self.train_P))))

    def __getitem__(self, idx):
        normal_path = os.path.join(self.root_dir, self.train_N, os.listdir(os.path.join(self.root_dir, self.train_N))[idx])
        pneumo_path = os.path.join(self.root_dir, self.train_P, os.listdir(os.path.join(self.root_dir, self.train_P))[idx])
        
        normal_img = Image.open(normal_path).convert("L")  # Load as grayscale
        pneumo_img = Image.open(pneumo_path).convert("L")  # Load as grayscale
        
        normal_img = self.transforms(normal_img)
        pneumo_img = self.transforms(pneumo_img)
        
        return normal_img, pneumo_img


