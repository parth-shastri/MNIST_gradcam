# Project Title: MNIST Digit Recognition with Grad-CAM

## Description

This project demonstrates the implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) for the MNIST dataset. Grad-CAM is a technique that helps visualize which parts of an image are contributing the most to a deep learning model's prediction.

In this project, we use a pre-trained model on the MNIST dataset to recognize handwritten digits. We then use Grad-CAM to highlight the regions in the input image that the model focuses on to make its prediction. The implementation includes both uploading images and drawing digits on a canvas for recognition and visualization.

Paper - [Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization](https://arxiv.org/pdf/1610.02391.pdf)

## Table of Contents

- [Installation](#installation)

- [Usage](#usage)

- [Demo](#demo)

## Installation

1\. Clone the repository:

   

    git clone https://github.com/your-username/mnist-grad-cam.git
    
    cd mnist-grad-cam

   

2\. Create the environment and install the required dependencies:

   
    
    python venv -n /path/to/environment
  
    source /path/to/environment/activate
  
    pip install -r requirements.txt

   

3\. Train your own model by making changes to model.py and run train.py:

    python train.py.

## Usage

1\. Directly run the Streamlit app to interact with the project:


    streamlit run main.py


2\. Open your web browser and navigate to the provided local URL (usually http://localhost:8501).

3\. Use the app to:

   - Upload an image of a handwritten digit for recognition and Grad-CAM visualization.
   
   - Draw any digit in freeform using the canvas on the UI.
     
   - See the recognized digit along with the Grad-CAM heatmap.

## Demo

TODO
