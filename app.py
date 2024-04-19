import streamlit as st
import torch
from torch import nn, optim
import torchvision.transforms.v2 as v2
import os
from PIL import Image
import shutil

st.markdown("<h1 style='text-align: center; color: white;'>Real vs. Fake Image Classification</h1>", unsafe_allow_html=True)

def get_image_path(img):
    # Create a directory and save the uploaded image.
    file_path = f"data/uploadedImages/{img.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(img.getbuffer())
    return file_path

class CustModel(torch.nn.Module):

    def __init__(self):
        """Constructor for the neural network."""
        super(CustModel, self).__init__()        # Call superclass constructor

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(in_features=512, out_features=2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.smalldrop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.smalldrop(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.smalldrop(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        return x

def preprocess(image) -> torch.Tensor:
  """
  Preprocesses an image by applying a series of transformation.

  Args:
      image (npt.ArrayLike): The input image to be preprocessed.

  Returns:
      torch.Tensor: The preprocessed image as a tensor.
  """
  mean = torch.tensor([0.5212, 0.4260, 0.3811])
  std = torch.tensor([0.2780, 0.2530, 0.2538])
  tensor_converter = v2.Compose([ # Step 0: Convert from PIL Image to Torch Tensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
  ])
  normalizer = v2.Normalize(mean=mean, std=std)
  grayer = v2.Grayscale(3)

  preprocessor = v2.Compose([
    tensor_converter,
    normalizer,
    grayer,
  ])
  return preprocessor(image)

def eval_image(image):
    st.columns(3)[1].image(image)
    image_path = get_image_path(image)
    image = Image.open(image_path)
    image = preprocess(image)
    # st.write(image)
    image = image.unsqueeze(0)
    model.eval()
    output = None
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        output = predicted
    shutil.rmtree("data")
    return output

model = CustModel()
model.load_state_dict(torch.load("rvf_cnn.pt", map_location=torch.device('cpu')))

image = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])

if image is not None:
    label = eval_image(image)[0] # 0 for fake, 1 for real
    if label == 1:
        st.markdown("<h2 style='text-align: center; color: white;'>Real!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: white;'>Fake!</h2>", unsafe_allow_html=True)


