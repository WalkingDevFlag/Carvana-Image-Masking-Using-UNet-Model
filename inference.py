import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
    return output

def display_results(image_path, prediction):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(prediction, cmap='gray')
    
    plt.show()

# Load your trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

# Test on a single image
image_path = "/kaggle/input/carvana-image-masking-png/train_images/00087a6bd4dc_01.jpg"  
image_tensor = load_image(image_path)
prediction = predict(model, image_tensor, device)

# Display the results
display_results(image_path, prediction)