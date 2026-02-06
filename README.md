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

# Experiment 4 – Data Augmentation & CNN Training 🧠🖼️

- A Python-based experiment that demonstrates data augmentation techniques to improve image classification performance.
- Uses PyTorch and Torchvision to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
- Compares model performance with and without data augmentation using accuracy curves.

## Objective

- To understand how data augmentation helps a deep learning model generalize better.
- To apply multiple augmentation techniques and observe their impact on CNN training accuracy.

## Augmentation Techniques Used

- Rotation – Rotates images randomly
- Translation – Shifts images horizontally and vertically
- Scaling – Zooms images in and out
- Image Addition – Increases brightness
- Image Subtraction – Enhances edges using Gaussian blur subtraction

## Features

- Train CNN without augmentation (original images only)
- Train CNN with augmentation (modified images)
- Plot and compare accuracy curves
- Automatic dataset download (CIFAR-10)

## Dataset

- CIFAR-10
- 60,000 color images
- Image size: 32 × 32
- 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images, 10,000 testing images



## Observe:

- Training accuracy without augmentation
- Training accuracy with augmentation
- Accuracy comparison graph


## Conclusion

- Models trained with data augmentation learn more robust features.
- Augmentation improves generalization and reduces overfitting.
- Accuracy curves clearly show the difference in learning behavior.
---
# Experiment 5 – Histogram Operations 📊

## Features

- Grayscale Histogram Equalization
- Improves the contrast of a grayscale image.
- Makes dark or bright images more balanced.
- Color Histogram Equalization
- Applies histogram equalization on RGB channels individually.
- Applies histogram equalization on HSV color space (enhancing brightness without changing colors).
- Histogram Matching (Specification)
- Adjusts one image to match the brightness/contrast of a reference image.
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhances contrast locally in small image regions.
- Prevents over-amplification of noise (better than global histogram equalization).

## Output

- Original Grayscale Image
- Grayscale Histogram Equalized Image
- CLAHE Grayscale Image
- Original Color Image
- RGB Histogram Equalized Image
- HSV Histogram Equalized Image
- Reference Image (for histogram matching)
- Histogram Matched Image

---

# Experiment 6 – Effect of Histogram Equalization on CNN Classification 🧠🖼️

- A deep learning experiment implemented using **PyTorch** to study the impact of **Histogram Equalization** on image classification.
- A **Simple Convolutional Neural Network (CNN)** is trained on the **CIFAR-10 dataset**.
- Model performance is compared **with and without histogram equalization**.
- This experiment helps understand the importance of **image preprocessing** in deep learning.


## Features

- **Training Without Histogram Equalization**
  - Images are converted to grayscale.
  - Pixel values are normalized.
  - No contrast enhancement is applied.

- **Training With Histogram Equalization**
  - Images are converted to grayscale.
  - Histogram equalization is applied to improve image contrast.
  - Pixel values are normalized before training.

- **CNN-Based Image Classification**
  - Uses a simple CNN architecture with convolution, pooling, and fully connected layers.
  - Trained using Cross-Entropy loss and Adam optimizer.

- **Accuracy Comparison**
  - Model accuracy is calculated for both cases.
  - Results are compared to analyze the effect of histogram equalization.

## Dataset

- **CIFAR-10 Dataset**
  - 60,000 images (32×32 pixels)
  - 10 different object classes
  - Dataset is automatically downloaded using torchvision

## Usage

- Open the notebook
- The CIFAR-10 dataset is downloaded automatically.
- The model is trained:
  - Once without histogram equalization
  - Once with histogram equalization
- Accuracy for both cases is printed.
- Compare the results to observe performance differences.

## Output

- Accuracy without Histogram Equalization
- Accuracy with Histogram Equalization
- Comparison shows how contrast enhancement affects CNN performance.

  
## Conclusion
Histogram equalization can improve or degrade performance depending on the dataset and model.

---
# Experiment 7 – Edge Detection Using Sobel Operator 🖼️📐

An image processing experiment implemented using **Python and OpenCV** to analyze **edge detection** using the **Sobel operator**.  
This experiment studies the effect of **kernel size variation (3×3 vs 5×5)** and **image smoothing** on edge detection performance.


## Objective

- To detect edges using the Sobel operator
- To analyze the effect of kernel size (3×3 and 5×5)
- To study edge detection before and after image smoothing
- To understand the importance of preprocessing in image enhancement


## Features

### Grayscale Conversion
- Converts the input image to grayscale
- Reduces complexity by removing color information

### Sobel Edge Detection (3×3 Kernel)
- Detects fine and thin edges
- More sensitive to noise
- Captures detailed edge information

### Sobel Edge Detection (5×5 Kernel)
- Produces thicker and smoother edges
- Less sensitive to noise
- Suitable for strong edge detection

### Gaussian Image Smoothing
- Reduces noise using Gaussian Blur
- Minimizes false edges

### Edge Detection After Smoothing
- Applies Sobel operator on smoothed image
- Produces clearer and more continuous edges

### Visualization
- Displays all outputs using Matplotlib subplots for easy comparison


## Input

- Any standard image (JPG / PNG)
- Image is uploaded manually (Google Colab) or loaded locally
- Image is converted to grayscale before processing

## Usage

- Open the Python notebook or script
- Upload the input image
- The program performs:
  - Sobel edge detection using 3×3 kernel
  - Sobel edge detection using 5×5 kernel
  - Gaussian smoothing
  - Sobel edge detection after smoothing
- Outputs are displayed for comparison

## Output

- Original grayscale image
- Sobel edge detection (3×3 kernel)
- Sobel edge detection (5×5 kernel)
- Smoothed (Gaussian blurred) image
- Sobel edge detection after smoothing

## Observations

- 3×3 kernel detects fine edges but is sensitive to noise
- 5×5 kernel produces smoother and thicker edges
- Smoothing improves edge continuity and reduces false edges

## Conclusion

This experiment demonstrates that kernel size and preprocessing significantly affect edge detection.  
Applying Gaussian smoothing before Sobel edge detection improves edge clarity and continuity, making it more suitable for image enhancement tasks.

---
# Experiment 8– Laplacian and Laplacian of Gaussian (LoG) Edge Detection 🖼️📐

An image processing experiment implemented using **Python and OpenCV** to analyze **edge detection** using the **Laplacian operator** and **Laplacian of Gaussian (LoG)**.  
The experiment compares edge detection results **with and without Gaussian smoothing** to study noise reduction and edge continuity.

## Objective

- To perform edge detection using the Laplacian operator
- To study the effect of noise on Laplacian edge detection
- To apply Gaussian smoothing before Laplacian
- To analyze the improvement using Laplacian of Gaussian (LoG)

## Features

### Grayscale Conversion
- Converts the input image to grayscale
- Simplifies processing by using intensity values only

### Laplacian Edge Detection
- Detects edges in all directions
- Sensitive to noise

### Gaussian Image Smoothing
- Applies Gaussian blur to reduce noise
- Prevents false edge detection

### Laplacian of Gaussian (LoG)
- Combines Gaussian smoothing and Laplacian edge detection
- Produces smoother and more continuous edges
- Reduces noise effects

### Visualization
- Displays results using Matplotlib subplots for comparison

## Input

- Any standard image (JPG / PNG)
- Image is uploaded manually (Google Colab) or loaded locally
- Converted to grayscale before processing

## Usage

- Open the Python notebook or script
- Upload the input image
- The program performs:
  - Grayscale conversion
  - Laplacian edge detection
  - Gaussian smoothing
  - Laplacian of Gaussian (LoG)
- Outputs are displayed for visual comparison


## Output

- Original grayscale image
- Edge detection using Laplacian
- Gaussian smoothed image
- Edge detection using Laplacian of Gaussian (LoG)


## Observations

- Laplacian detects sharp edges but is highly sensitive to noise
- Gaussian smoothing reduces noise in the image
- LoG produces clearer, smoother, and more continuous edges
- LoG performs better than plain Laplacian for noisy images


## Conclusion

This experiment shows that applying Gaussian smoothing before Laplacian edge detection significantly improves edge quality.  
Laplacian of Gaussian (LoG) is more effective than Laplacian alone due to reduced noise sensitivity and better edge continuity.

---

# Experiment 9 – Multi-Layer Neural Network (MLNN) From Scratch 🧠📉

A Python-based experiment that implements a Multi-Layer Neural Network (MLNN) from scratch without using any inbuilt machine learning or deep learning libraries.
The experiment demonstrates forward propagation, backpropagation, loss calculation, weight updates, and decision boundary visualization using a simple binary classification problem (XOR dataset).

## Objective

- To understand the working of a Multi-Layer Neural Network
- To implement forward propagation manually
- To implement backward propagation manually
- To study the effect of epochs on loss reduction
- To visualize the decision boundary / clustering diagram
- To analyze how a neural network learns a non-linear problem (XOR)

## Network Architecture

- Input Layer: 2 neurons (X₁, X₂)
- Hidden Layer: 2 neurons
- Activation Function: ReLU
- Output Layer: 1 neuron
- Activation Function: Sigmoid

## Dataset

- XOR Dataset

Input:
- (0, 0) → 0
- (0, 1) → 1
- (1, 0) → 1
- (1, 1) → 0

## Hyperparameters
- Learning Rate: 0.1
- Epochs: 1000
- Activation Functions:
  - ReLU (Hidden Layer)
  - Sigmoid (Output Layer

## Observations

- Loss decreases gradually as epochs increase
- The network successfully learns a non-linear decision boundary
- ReLU helps the network learn complex patterns
- Sigmoid produces probability-based output suitable for binary classification


