import numpy as np
import os
from django.conf import settings
from pathlib import Path
from tensorflow.keras.models import load_model
from .aiutils import InstanceNormalization, SpectralNormalization, residual_block
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

DEF_PATH = Path(__file__).resolve().parent

# from skimage.metrics import structural_similarity as ssim

# def calculate_mse_loss(clean_image, denoised_image):
#     # MSE Loss
#     return np.mean((clean_image - denoised_image) ** 2)

# def calculate_ssim_loss(clean_image, denoised_image):
#     # SSIM Loss (higher SSIM = better quality, so we use 1 - SSIM as "loss")
#     return 1 - ssim(clean_image, denoised_image, multichannel=True)

# def calculate_accuracy(clean_image, denoised_image, tolerance=5):
#     # Accuracy: Percentage of pixels with differences less than the tolerance
#     pixel_diff = np.abs(clean_image - denoised_image)
#     accurate_pixels = np.sum(pixel_diff < tolerance)
#     total_pixels = clean_image.size
#     return (accurate_pixels / total_pixels) * 100


def model():
    # Define custom objects for loading
    custom_objects = {
        "InstanceNormalization": InstanceNormalization,
        "SpectralNormalization": SpectralNormalization,
        "residual_block": residual_block
    }
    model = load_model(os.path.join(DEF_PATH, 'models',
                       'model.keras'), custom_objects=custom_objects)
    return model



def convert(input_image_url, output_image_path, resolution=None):
    """
    Converts a noisy image to a denoised image using the trained GAN model.

    Parameters:
    - input_image_url (str): Path to the noisy input image.
    - output_image_path (str): Path where the denoised image will be saved.
    - model_path (str): Path to the trained model file.
    - resolution (str, optional): Target resolution for the output image (e.g., "512x512").
    
    Returns:
    - success (bool): True if image is saved successfully, False otherwise.
    - mse_loss (float): Placeholder for MSE loss computation (not implemented yet).
    - accuracy (float): Placeholder for accuracy computation (not implemented yet).
    """

    try:
        # Load and preprocess the noisy input image (same as training)
        img = img_to_array(load_img(input_image_url, target_size=(256, 256))) / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Load model
        loaded_model = model()

        # Predict denoised image
        denoised_output = loaded_model.predict(img)[0]  # Shape: (256, 256, 3)

        # Convert back to PIL image format
        denoised_image = array_to_img(denoised_output)

        # Resize to original or specified resolution
        if resolution:
            try:
                w, h = map(int, resolution.split('x'))
                denoised_image = denoised_image.resize((w, h))
            except Exception as e:
                print(f"Error resizing image: {e}")

        # Save the denoised image using PIL
        denoised_image.save(output_image_path)
        print(f"Saved: {output_image_path}")

        mse_loss, accuracy = 0, 0  # Placeholder values
        return True, mse_loss, accuracy

    except Exception as e:
        print(f"Error processing image: {e}")
        return False, 0, 0  # Return failure


def initiate_conversion(photo_conversion):
    input_image_url = os.path.join(
        settings.MEDIA_ROOT, photo_conversion.input_image.name)  # .replace("/", "\\")
    file_extension = photo_conversion.input_image.name.split('.')[-1]
    file_name = f'{photo_conversion.reference_id}.{file_extension}'
    output_image_path = os.path.join(
        settings.MEDIA_ROOT, 'output_images', file_name)  # .replace("/", "\\")
    converted, loss, accuracy = convert(
        input_image_url, output_image_path, photo_conversion.resolution)
    if converted:
        photo_conversion.status = 'completed'
        photo_conversion.output_image = "/output_images/" + file_name
        photo_conversion.accuracy = accuracy
        photo_conversion.loss = loss
        photo_conversion.save()


def convert_all(from_dir, to_dir):
    files = os.listdir(from_dir)
    # print(files)
    for f in files:
        from_path = os.path.join(from_dir, f)
        to_path = os.path.join(to_dir, f)
        # if os.path.isfile(f):
        convert(from_path, to_path)
        # print(f"converting {from_path} to {to_path}")
        # else:
        #     print(
        #         f"Skipping {from_path} | {to_path} because it is not a file.")
