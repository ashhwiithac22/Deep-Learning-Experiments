# Experiment 1 – RGB Channels & Binary Image 🖼️

- A Python-based experiment that allows users to upload an image, visualize its **Red, Green, and Blue channels** separately, and convert it into a **binary (black & white) image** using thresholding.
- Built with **OpenCV** for image manipulation and **Matplotlib** for visualization.

## Features

- **Image Upload**: Upload any image directly from your device.
- **RGB Channel Visualization**: View Red, Green, and Blue channels individually.
- **Binary Image Conversion**: Convert the image to black & white using thresholding.

## Setup

```bash
pip install opencv-python matplotlib numpy google-colab
```
## Usage

- Open DL_Ex1.ipynb in Google Colab.
- Upload an image when prompted.
- Visualize the original image, RGB channels, and binary image results.

--- 

# Experiment 2 – Geometric Transformations 🔄
- A Python-based experiment demonstrating rotation, translation, scaling, and flipping of images using OpenCV.
- Designed to show how images can be geometrically modified.

## Features
- **Rotation**: Rotate an image by 30°, 60°, and 90°.
- **Translation**: Shift an image horizontally and vertically.
- **Scaling** : Enlarge or shrink the image.
- **Flipping**: Flip the image horizontally, vertically, or both.

## Setup
```
pip install opencv-python matplotlib numpy google-colab
```

# Experiment 3 – Pixel-Level Operations 🎨

- A Python-based experiment performing image addition (blending) and image subtraction (change detection) using pixel-wise operations in OpenCV.
- Helps understand how pixel intensity values interact between images.

## Features

- **Image Addition (Blending)**: Mix a grayscale version of the image with the original RGB image.
- **Image Subtraction** : Compare two images to highlight movement or brightness differences.
- **Auto-Generate Second Image** : If only one image is uploaded, a shifted copy is created for subtraction.

## Setup
```
pip install opencv-python matplotlib numpy google-colab
```

## Usage

- Open DL_Ex3.ipynb in Google Colab.
- Upload one or two images when prompted.
- View blended results and subtraction-based difference detection.


--- 
