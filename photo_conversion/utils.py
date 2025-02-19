import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
# from keras.models import model_from_json
import os
from django.conf import settings
from pathlib import Path

DEF_PATH = Path(__file__).resolve().parent

from skimage.metrics import structural_similarity as ssim

def calculate_mse_loss(clean_image, denoised_image):
    # MSE Loss
    return np.mean((clean_image - denoised_image) ** 2)

def calculate_ssim_loss(clean_image, denoised_image):
    # SSIM Loss (higher SSIM = better quality, so we use 1 - SSIM as "loss")
    return 1 - ssim(clean_image, denoised_image, multichannel=True)

def calculate_accuracy(clean_image, denoised_image, tolerance=5):
    # Accuracy: Percentage of pixels with differences less than the tolerance
    pixel_diff = np.abs(clean_image - denoised_image)
    accurate_pixels = np.sum(pixel_diff < tolerance)
    total_pixels = clean_image.size
    return (accurate_pixels / total_pixels) * 100


def model():
    from tensorflow.keras.models import load_model
    try:
        model = load_model(os.path.join(DEF_PATH, 'models', 'model.h5'))
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def convert(input_image_url, output_image_path, resolution=None):
    from skimage.metrics import structural_similarity as ssim

    try:
        # Load and preprocess the noisy input image
        img = cv2.imread(input_image_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (256, 256))

        # Load model and predict denoised image
        denoise_array = np.array([img_resized], dtype=float) / 255.0
        loaded_model = model()
        denoised_output = loaded_model.predict(denoise_array)[0]  # Shape: (256, 256, 3)

        # Scale back to image range [0, 255]
        denoised_image = (denoised_output * 255).astype(np.uint8)

        # # If ground truth (clean image) is available, calculate loss and accuracy
        # clean_image_path = input_image_url.replace("gussian", "clear")  # Assuming clear image naming convention
        # clean_image = cv2.imread(clean_image_path)
        # clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        # clean_image_resized = cv2.resize(clean_image, (256, 256))

        # # Calculate loss and accuracy
        # mse_loss = calculate_mse_loss(clean_image_resized, denoised_image)
        # ssim_loss = calculate_ssim_loss(clean_image_resized, denoised_image)
        # accuracy = calculate_accuracy(clean_image_resized, denoised_image)

        mse_loss, accuracy = 0,0
        # Save the denoised image
        success = cv2.imwrite(output_image_path, cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))
        return success, mse_loss, accuracy
    except Exception as e:
        print(f"Error in convert function: {e}")
        return False, None, None

def initiate_conversion(photo_conversion):
    input_image_url = os.path.join(
        settings.MEDIA_ROOT, photo_conversion.input_image.name)  # .replace("/", "\\")
    output_image_path = os.path.join(
        settings.MEDIA_ROOT, 'output_images', photo_conversion.reference_id + '.jpg')  # .replace("/", "\\")
    converted, loss, accuracy = convert(
        input_image_url, output_image_path, photo_conversion.resolution)
    if converted:
        photo_conversion.status = 'completed'
        photo_conversion.output_image = "/output_images/" + \
            photo_conversion.reference_id + ".jpg"
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
