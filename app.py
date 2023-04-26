import io
import torch
import torch.nn as nn
from torchvision import transforms
import streamlit as st
from PIL import Image
# Craete a neural network from pytorch
# https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(3*3*64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


img_size = 224

# Define the transformation for the training set, with data augmentation
transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Label mapping
label_map = {"nv": 0, "mel": 1, "bkl": 2,
             "bcc": 3, "akiec": 4, "vasc": 5, "df": 6}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


def main():
    st.title("Skin Cancer Detection")
    st.write("This is a simple image classification web app to predict skin cancer")
    # Add a footer
    st.write(
        "Built with Streamlit by [Auston](https://www.linkedin.com/in/austonpramodh) and [Prajakta](https://www.linkedin.com/in/prajakta-chaudhari/)"
    )

    # Create an instance of the model
    model = CNN(num_classes=7)
    # Load the trained model parameters
    model.load_state_dict(torch.load("model.pth"))
    # Set the model in evaluation mode
    model.eval()

    # Create a file uploader widget
    file_uploader = st.file_uploader("Upload an image", type=["png", "jpg"])

    if file_uploader is not None:
        # Read the image
        image = file_uploader.read()
        # Convert the image to RGB format
        image = Image.open(io.BytesIO(image)).convert("RGB")
        # print(type(image))
        # Transform the image to tensor
        image = transformer(image).unsqueeze(0)
        # Make predictions
        predictions = model(image)
        # Get the predicted class with the highest score
        prediction = predictions.argmax(dim=1)
        # Get the class label
        label = prediction.item()
        # Map the label
        label = list(label_map.keys())[list(label_map.values()).index(label)]
        lesson = lesion_type_dict[label]
        # Show the image
        st.image(file_uploader, use_column_width=True)
        # Print the class label
        st.write(f"Predicted class: {label}")
        st.write(f"Lesion type: {lesson}")


if __name__ == "__main__":
    main()
