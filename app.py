import streamlit as st
import os
import cv2 as cv
import torch
from torchvision import transforms
from torch.optim import LBFGS
from torch.autograd import Variable
from models.definitions.vgg19 import Vgg19
import numpy as np

# Constants for normalizing images using ImageNet mean and std deviation.
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]
    
    # Extract features from the current image.
    current_set_of_feature_maps = neural_net(optimizing_img)
    
    # Content loss: Mean squared error between target and current content representations.
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(0)
    content_loss = torch.nn.MSELoss()(target_content_representation, current_content_representation)
    
    # Style loss: Sum of MSE losses between target and current style Gram matrices.
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for x in current_set_of_feature_maps]
    for target_gram, current_gram in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss()(target_gram, current_gram)
    style_loss /= len(target_style_representation)
    
    # Total variation loss for smoothness.
    tv_loss = torch.sum(torch.abs(optimizing_img[:, :, :, :-1] - optimizing_img[:, :, :, 1:])) + \
              torch.sum(torch.abs(optimizing_img[:, :, :-1, :] - optimizing_img[:, :, 1:, :]))
    
    # Total loss is a weighted sum of content, style, and TV loss.
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss
    
    return total_loss, content_loss, style_loss, tv_loss

def load_image(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)[:, :, ::-1]  # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0
    return img

def prepare_img(img, target_shape, device):
    img = cv.resize(img, (target_shape, target_shape), interpolation=cv.INTER_CUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
    img = transform(img).to(device).unsqueeze(0)
    return img

def save_image(tensor, img_path):
    img = tensor.clone().cpu().squeeze(0)
    img = img.permute(1, 2, 0).detach().numpy()
    img = np.clip(img + np.array(IMAGENET_MEAN_255), 0, 255).astype('uint8')
    cv.imwrite(img_path, img[:, :, ::-1])

def neural_style_transfer(content_img, style_img, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content_img = prepare_img(content_img, config['height'], device)
    style_img = prepare_img(style_img, config['height'], device)
    
    # Start the optimization with a random image (or use content_img for better structure retention).
    optimizing_img = torch.randn(content_img.data.size(), device=device).mul_(0.5)  # Initialize with random noise
    optimizing_img = Variable(optimizing_img, requires_grad=True)
    
    neural_net = Vgg19(requires_grad=False).to(device).eval()
    content_feature_maps_index = neural_net.content_feature_maps_index
    style_feature_maps_indices = neural_net.style_feature_maps_indices
    
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index].squeeze(0)
    target_style_representation = [gram_matrix(x) for x in style_img_set_of_feature_maps]
    
    optimizer = LBFGS([optimizing_img], max_iter=config['iterations'])
    
    def closure():
        optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net, optimizing_img, [target_content_representation, target_style_representation], 
            content_feature_maps_index, style_feature_maps_indices, config
        )
        total_loss.backward()
        return total_loss
    
    for i in range(config['iterations']):
        optimizer.step(closure)
        st.progress((i + 1) / config['iterations'])
    
    return optimizing_img

# Center the UI using columns.
st.title("Neural Style Transfer")

with st.columns([1, 2, 1])[1]:
    st.sidebar.header("Configuration")

    # Upload content and style images.
    content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    # Allow users to adjust weights and iterations.
    content_weight = st.sidebar.slider("Content Weight", 1e3, 1e5, 1e5, step=1e3)
    style_weight = st.sidebar.slider("Style Weight", 1e3, 1e5, 3e4, step=1e3)
    tv_weight = st.sidebar.slider("Total Variation Weight", 0.0, 10.0, 1.0)
    iterations = st.sidebar.slider("Iterations", 10, 500, 200)

    # Set image size.
    image_size = st.sidebar.slider("Image Size (px)", 200, 800, 400)

    # When the button is pressed, start the process.
    if st.sidebar.button("Start Style Transfer"):
        if content_image and style_image:
            # Load and process the images.
            content_img = load_image(content_image)
            style_img = load_image(style_image)
            
            config = {
                'content_weight': content_weight,
                'style_weight': style_weight,
                'tv_weight': tv_weight,
                'height': image_size,
                'iterations': iterations
            }
            
            # Perform neural style transfer.
            result_img = neural_style_transfer(content_img, style_img, config)
            
            # Save and display the result.
            output_path = "output_image.jpg"
            save_image(result_img, output_path)
            
            st.success("Styling complete!")
            st.image(output_path, caption="Styled Image", use_column_width=True)
        else:
            st.error("Please upload both content and style images.")
