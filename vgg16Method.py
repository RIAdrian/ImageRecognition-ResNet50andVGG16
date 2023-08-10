# Import necessary libraries
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import os
from torchvision import models, transforms
import torch
import json

# Download the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Define image preprocessing transformations to match VGG16 requirements
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class names corresponding to ImageNet labels
with open("imagenet_classes.json") as f:
    class_names = json.load(f)

# Folder containing downloaded images
folder_path = "downloaded_images"

# PDF file name to be created
pdf_filename = "vgg16_images.pdf"

# Page dimensions (letter size)
page_width, page_height = letter

# Create a PDF file using ReportLab
c = canvas.Canvas(pdf_filename, pagesize=(page_width, page_height))

# Initial y-coordinate for placing the first image
y_coordinate = page_height - 50

# Iterate through images in the folder
for image_name in os.listdir(folder_path):
    if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            image_path = os.path.join(folder_path, image_name)
            img = Image.open(image_path)

            # Convert the image to RGB color mode
            img = img.convert("RGB")

            # Resize the image to fit within the page (if needed)
            img_width, img_height = img.size
            if img_width > page_width - 100:
                img_height = int((page_width - 100) * img_height / img_width)
                img_width = page_width - 100

            # Draw the image onto the PDF canvas
            c.drawImage(image_path, 50, y_coordinate - img_height, width=img_width, height=img_height)

            # Apply transformations to prepare the image for the neural network
            img_tensor = preprocess(img)

            # Run the image through the neural network
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0))
            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_names[predicted_idx.item()]

            # Add the label above the image in the PDF
            c.drawString(50, y_coordinate + 10, "Label: " + predicted_class)

            # Update the y-coordinate for the next image
            y_coordinate -= img_height + 60

            # Check if the next image exceeds the current page limit and add a new page if needed
            if y_coordinate < 50:
                c.showPage()
                y_coordinate = page_height - 50

        except Exception as e:
            print(f"Image {image_name} encountered an error: {e}")

# Save and close the PDF file
c.save()

# Print success message
print("PDF file successfully created:", pdf_filename)
