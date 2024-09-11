import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from models.definitions.vgg19 import Vgg19

# Constants for normalizing images using ImageNet mean and std deviation.
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path, target_shape="None"):
    '''
    Load an image from a file and optionally resize it.
    '''
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # Read image and convert from BGR to RGB.
    if target_shape is not None:
        # Resize image based on given target shape (height or tuple).
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img_path, target_shape, device):
    '''
    Load, resize, and normalize the image for the model.
    '''
    img = load_image(img_path, target_shape=target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor.
        transforms.Lambda(lambda x: x.mul(255)),  # Multiply pixel values by 255.
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])  # Normalize using ImageNet mean.
    img = transform(img).to(device).unsqueeze(0)  # Add batch dimension and move to the desired device (GPU/CPU).
    return img

def save_image(img, img_path):
    '''
    Save the generated image back to disk.
    '''
    if len(img.shape) == 2:  # If image is grayscale, replicate it across RGB channels.
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])  # Save image and convert RGB to BGR format.

def generate_out_img_name(config):
    '''
    Generate a name for the output image based on the content and style images.
    '''
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    suffix = f'{config["img_format"][1]}'  # Use the format suffix (e.g., '.jpg').
    return prefix + suffix

def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations):
    '''
    Save the intermediate or final image during the optimization process.
    '''
    saving_freq = -1  # Frequency for saving images (-1 means save only final image).
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()  # Detach image tensor and convert to numpy.
    out_img = np.moveaxis(out_img, 0, 2)  # Rearrange axes to get (H, W, C).

    # Save only the final image.
    if img_id == num_of_iterations - 1:
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))  # Add back the mean for de-normalization.
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')  # Clip pixel values to valid range.
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])  # Save as image file.

def prepare_model(device):
    '''
    Load a pre-trained VGG19 model for feature extraction.
    '''
    model = Vgg19(requires_grad=False, show_progress=True)
    # Indices of layers from which content and style features will be extracted.
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names  # Return the model in eval mode.

def gram_matrix(x, should_normalize=True):
    '''
    Compute the Gram matrix for a given feature map (used for style representation).
    '''
    (b, ch, h, w) = x.size()  # Batch size, channels, height, width.
    features = x.view(b, ch, w * h)  # Reshape to (B, C, H*W).
    features_t = features.transpose(1, 2)  # Transpose to (B, H*W, C).
    gram = features.bmm(features_t)  # Compute Gram matrix by multiplying features with its transpose.
    if should_normalize:
        gram /= ch * h * w  # Normalize by the number of elements.
    return gram

def total_variation(y):
    '''
    Compute the total variation loss, which promotes spatial smoothness in the generated image.
    '''
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Compute the total loss as a weighted sum of content, style, and total variation losses.
    '''
    target_content_representation = target_representations[0]  # Content features.
    target_style_representation = target_representations[1]  # Style features.
    current_set_of_feature_maps = neural_net(optimizing_img)  # Extract features from the current image.
    # Content loss: Mean squared error between target and current content representations.
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    # Style loss: Sum of MSE losses between target and current style Gram matrices.
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    # Total variation loss.
    tv_loss = total_variation(optimizing_img)

    # Total loss is a weighted sum of content, style, and TV loss.
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Perform a single step of the optimization (updating the pixels in the image).
    '''
    def tuning_step(optimizing_img):
        # Compute total loss and backpropagate.
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net, 
            optimizing_img, 
            target_representations, 
            content_feature_maps_index, 
            style_feature_maps_indices, 
            config
        )

        total_loss.backward()  # Compute gradients.
        optimizer.step()  # Update the pixels.
        optimizer.zero_grad()  # Clear gradients.
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def neural_style_transfer(config):
    '''
    Main function to perform neural style transfer.
    '''
    # Paths to content and style images.
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    
    # Create a directory to save the generated images.
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    
    # Set device to GPU (if available) or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare content and style images.
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    # Initialize the generated image with the content image.
    init_img = content_img
    optimizing_img = Variable(init_img, requires_grad=True)  # Make the image trainable.
    
    # Prepare VGG19 model and extract feature layers for content and style.
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
    
    # Extract content and style representations from the original images.
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    
    target_representations = [target_content_representation, target_style_representation]  # Store target representations.
    
    num_of_iterations = 200  # Number of iterations for the optimization.

    # Use the L-BFGS optimizer (commonly used for style transfer).
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    
    cnt = 0  # To track iterations.

    def closure():
        '''
        Closure function required by L-BFGS optimizer, evaluates the loss.
        '''
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()  # Clear gradients.
        
        # Compute total loss and backpropagate.
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()  # Backpropagate if required.
        
        with torch.no_grad():
            # Log the loss values and save the generated image at the final iteration.
            print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)
        
        cnt += 1
        return total_loss
    
    optimizer.step(closure)  # Perform optimization using the closure function.
    return dump_path  # Return the path where results are saved.

# Set paths to input/output images and configuration for style transfer.
PATH = ''
CONTENT_IMAGE = 'c3.jpg'
STYLE_IMAGE = 's2.jpg'

default_resource_dir = os.path.join(PATH, 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')
img_format = (4, '.jpg')

# Define the weights for content, style, and total variation loss.
optimization_config = {'content_img_name': CONTENT_IMAGE, 'style_img_name': STYLE_IMAGE, 'height': 400, 'content_weight': 100000.0, 'style_weight': 30000.0, 'tv_weight': 1.0}
optimization_config['content_images_dir'] = content_images_dir
optimization_config['style_images_dir'] = style_images_dir
optimization_config['output_img_dir'] = output_img_dir
optimization_config['img_format'] = img_format

# Run the neural style transfer and get the result path.
results_path = neural_style_transfer(optimization_config)
