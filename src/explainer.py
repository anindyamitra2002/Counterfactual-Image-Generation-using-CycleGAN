import numpy as np
import cv2
import torchvision.transforms as transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[:, target_class].sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cam.squeeze()

        return cam

    def visualize_cam(self, cam, input_image):
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        input_image = np.moveaxis(input_image.cpu().numpy()[0], 0, -1)
        input_image = np.float32(input_image)
        cam_image = heatmap + input_image
        cam_image = cam_image / cam_image.max()
        return cam_image

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image
