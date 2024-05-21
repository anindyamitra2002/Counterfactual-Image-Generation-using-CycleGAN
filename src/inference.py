import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from classifier import Classifier
from models import ResUNetGenerator

# Ensure the necessary directories exist
os.makedirs('results/translated_N', exist_ok=True)
os.makedirs('results/translated_P', exist_ok=True)

def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize image to 512x512
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
    ])
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    # Check if the tensor is on the GPU and move it to the CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Remove batch dimension and convert to NumPy array
    array = tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    
    # Denormalize the image if it was normalized to [-1, 1]
    array = (array * 0.5 + 0.5) * 255
    
    # Ensure the array is of type uint8
    array = array.astype(np.uint8)
    
    # Handle grayscale images
    if array.shape[2] == 1:
        array = array.squeeze(2)
        image = Image.fromarray(array, mode='L')
    else:
        image = Image.fromarray(array)
    
    # Save the image
    image.save(path)

def generate_images(image_path, classifier_path, g_NP_checkpoint, g_PN_checkpoint, image_size=512):
    # Load the classifier model
    classifier = Classifier()
    classifier_checkpoint = torch.load(classifier_path, map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_checkpoint['state_dict'])
    classifier.eval()

    # Load the generator models
    g_NP = load_model(g_NP_checkpoint, ResUNetGenerator(gf=32, channels=1))
    g_PN = load_model(g_PN_checkpoint, ResUNetGenerator(gf=32, channels=1))

    # Load the input image
    image = load_image(image_path, image_size)

    # Classify the image
    classifier_output = classifier(image).cpu().detach().numpy()
    pred = np.argmax(classifier_output, axis=1)[0]
  
    if pred > 0.5:
        print("Classified as Domain P")
        translate_to_domain = g_PN
        folder_to_save = 'results/translated_N'
        reverse_translate = g_NP
    else:
        print("Classified as Domain N")
        translate_to_domain = g_NP
        folder_to_save = 'results/translated_P'
        reverse_translate = g_PN

    # Perform translation and save images
    with torch.no_grad():
        for i in range(10):  # Generate and save 10 images
            translated_image = translate_to_domain(image)
            save_image(translated_image, os.path.join(folder_to_save, f'translated_{i}.png'))

            # Translate back to the original domain and save
            recon_image = reverse_translate(translated_image)
            save_image(recon_image, os.path.join(folder_to_save, f'recon_{i}.png'))

if __name__ == "__main__":
    # Replace these paths with the correct paths to your files
    image_path = '0a2f6cf6-1f45-44c8-bcf0-98a3b466.png'
    classifier_path = 'models\efficientnet_b1-epoch16-val_loss0.46_ft.ckpt'
    g_NP_checkpoint = 'models\g_NP_best.ckpt'
    g_PN_checkpoint = 'models\g_PN_best.ckpt'

    generate_images(image_path, classifier_path, g_NP_checkpoint, g_PN_checkpoint)
