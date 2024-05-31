import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from src.classifier import Classifier

# Ensure the necessary directories exist
# os.makedirs('results/translated_N', exist_ok=True)
# os.makedirs('results/translated_P', exist_ok=True)

# Load the classifier model
def load_classifier(classifier_path):
    classifier = Classifier()
    classifier_checkpoint = torch.load(classifier_path, map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_checkpoint['state_dict'])
    classifier.eval()
    return classifier

# Load the generator models
def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_image(input_image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize image to 512x512
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
    ])
    input_image = input_image.convert('L')
    return transform(input_image).unsqueeze(0)

def convert_into_image(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    array = (array * 0.5 + 0.5) * 255
    array = array.astype(np.uint8)
    
    if array.shape[2] == 1:
        array = array.squeeze(2)
        image = Image.fromarray(array, mode='L')
    else:
        image = Image.fromarray(array)
    
    return image

def generate_images(input_image, classifier, g_PN, g_NP, image_size=512):

    image = load_image(input_image, image_size)

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
        for i in range(1):  # Generate and save 10 images
            translated_image = translate_to_domain(image)
            # save_image(translated_image, os.path.join(folder_to_save, f'translated_{i}.png'))

            # Translate back to the original domain and save
            recon_image = reverse_translate(translated_image)
            # save_image(recon_image, os.path.join(folder_to_save, f'recon_{i}.png'))

    return translated_image, recon_image

def classify_image(input_image, classifier, image_size=512):
    
    image = load_image(input_image, image_size)
    classifier_output = classifier(image).cpu().detach().numpy()
    pred = np.argmax(classifier_output, axis=1)[0]
    if pred > 0.5:
        return {"Pneumonia": classifier_output[0][1]}, 1
    
    else:
        return {"Normal": classifier_output[0][0]}, 0
