import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Class Names
# --------------------------------------------------
CLASS_NAMES = [
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
    "Tuberculosis"
]

# --------------------------------------------------
# Image Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Build DenseNet121
# --------------------------------------------------
def create_model():

    model = models.densenet121(
        weights=None
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 6)
    )

    return model


# --------------------------------------------------
# Load Model
# --------------------------------------------------
def load_model(model_path):

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


# --------------------------------------------------
# Predict
# --------------------------------------------------
def predict_image(model, image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "class": CLASS_NAMES[prediction.item()],
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze().cpu().numpy()
    }