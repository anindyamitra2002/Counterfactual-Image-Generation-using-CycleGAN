import gradio as gr


from src.inference import load_classifier, load_model, generate_images, convert_into_image, classify_image
from src.models import ResUNetGenerator
from src.explainer import GradCAM, preprocess_image

# Loading Models
classifier_path = 'models\\efficientnet_b1-epoch16-val_loss0.46_ft.ckpt'
g_NP_checkpoint = 'models\\g_NP_best.ckpt'
g_PN_checkpoint = 'models\\g_PN_best.ckpt'
g_NP = load_model(g_NP_checkpoint, ResUNetGenerator(gf=32, channels=1))
g_PN = load_model(g_PN_checkpoint, ResUNetGenerator(gf=32, channels=1))
classifier = load_classifier(classifier_path)
target_layer = classifier.model.features[-1]
grad_cam = GradCAM(classifier, target_layer)


def counterfactual_generation(input_image):

    translated_images, recon_images = generate_images(input_image, classifier, g_PN, g_NP)
    translated_images = convert_into_image(translated_images)
    recon_images = convert_into_image(recon_images)
    return translated_images, recon_images

def image_classification(input_image):

    result, target_class = classify_image(input_image, classifier=classifier)
    input_tensor = preprocess_image(input_image)
    cam = grad_cam.generate_cam(input_tensor, target_class)
    cam_image = grad_cam.visualize_cam(cam, input_tensor)

    return result, cam_image

# Defining the components
inputs1 = gr.Image(type="pil", format="png")
inputs2 = gr.Image(type="pil", format="png")
outputs1 = [gr.Image(type="pil", label="Translated Images", format="png"),
           gr.Image(type="pil", label="Reconstructed Images", format="png")]

outputs2 = [gr.Label(label="Classification Result"), gr.Image(label="Grad-CAM", format="png")]

with gr.Blocks() as demo:
    with gr.Tab("Counterfactual Generation"):
        app1 = gr.Interface(fn=counterfactual_generation, inputs=inputs1, outputs=outputs1,
                            title="Counterfactual Image Generation", allow_flagging="never",
                            description="Generate counterfactual images to explain the classifier's decisions.")

    with gr.Tab("Classification"):
        app2 = gr.Interface(fn=image_classification, inputs=inputs2, outputs=outputs2,
                            title="Image Classification", allow_flagging="never",
                            description="Classify the input medical image and visualize Grad-CAM.")

# Launch the app
demo.launch(share=True)