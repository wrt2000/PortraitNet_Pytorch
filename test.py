import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from models.PortraitNet import PortraitNet

def concatenate_images(image1, image2, image3, output_path):

    width1, height1 = image1.size
    width2, height2 = image2.size
    image2 = image2.resize((width1, height1), Image.BILINEAR)
    image3 = image3.resize((width1, height1), Image.BILINEAR)
    new_image = Image.new('RGB', (width1 * 3, height1))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    new_image.paste(image3, (width1 + width2, 0))

    new_image.save(output_path)


def load_model(model_path, device):
    model = PortraitNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    return model


def process_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image


def predict_mask(model, image, device):
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        mask, _ = model(image)
    mask = torch.sigmoid(mask)
    mask = torch.argmax(mask, dim=1)
    mask = mask.squeeze().cpu().numpy()
    return mask


def extract_portrait(original_image, mask):
    original_image = np.array(original_image)
    original_size = original_image.shape[:2]

    # Resize mask to the original image size
    mask = mask > 0.5  # Thresholding

    # Create a 3 channel mask for image multiplication
    mask_3channel = np.stack((mask,)*3, axis=-1)

    # Extract portrait with transparent background
    portrait = np.where(mask_3channel, original_image, [0, 0, 0])  # Replace background with black (for transparency)
    portrait = Image.fromarray(portrait.astype(np.uint8))

    # Convert black background to transparent
    portrait = portrait.convert("RGBA")
    newData = []
    for item in portrait.getdata():
        if item[0] == 0 and item[1] == 0 and item[2] == 0:  # Finding black color by RGB values
            newData.append((255, 255, 255, 0))  # Setting alpha to 0 for transparency
        else:
            newData.append(item)
    portrait.putdata(newData)

    return portrait



def main(image_path, model_path, device='cuda'):
    model = load_model(model_path, device)
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((224, 224), Image.BILINEAR)
    processed_image = process_image(image_path, 224)  # Assuming input size is 224
    mask = predict_mask(model, processed_image, device)
    # save mask
    portrait = extract_portrait(original_image, mask)
    mask = (mask * 255).clip(0, 255).astype(np.uint8)

    mask_image = Image.fromarray(mask)
    # mask_image.save('predicted_mask.png')
    # portrait.save('extracted_portrait.png')

    concatenate_images(original_image, mask_image, portrait, save_path)


# Example usage
# image_path = 'dataset/EG1800/Images/00005.png'
# model_path = 'results/eg1800_20231114161557/ckpt_epoch_1500.pth'
# save_path = '00005.png'

image_path = 'dataset/Supervisely_face/SuperviselyPerson_ds1/resize_img/entrepreneur-startup-start-up-man-39866.png'
model_path = 'results/Supervisely_face_20231113221803/ckpt_epoch_1500.pth'
save_path = 's3.png'

main(image_path, model_path)
