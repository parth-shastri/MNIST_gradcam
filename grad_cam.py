import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from model import SimpleMNISTClassifier
import os
import cv2


class GRADCAM_Model(nn.Module):
    def __init__(self, classifier, saved_path=None):
        super().__init__()
        # init the model 
        self.model = classifier
        if saved_path is not None:
            self.model.load_state_dict(torch.load(saved_path, map_location=torch.device('cpu')))
        
        self.first_layer_activations = self.model.layer_1[:]
        self.feature_activations = nn.Sequential(
            self.first_layer_activations,
            self.model.layer_2[:-1]
        )
        # print(self.feature_activations)

    def get_activation(self, x):
        return self.feature_activations(x)
    
    def get_activation_gradients(self):
        return self.gradients
    
    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, inputs):
        x = self.feature_activations(inputs)

        h = x.register_hook(self.activation_hook)

        x = self.model.layer_2[-1](x)
        x = x.reshape(x.shape[0], -1)
        x = self.model.fc_1(x)
        return x

def preprocess(img):
    # preprocess
    img = np.array(img)

    img = cv2.resize(img, (28, 28))

    # print(img.shape)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)

    if img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = np.expand_dims(img, axis=-1)
    
    assert img.shape == (28, 28, 1)
    return img
    

def gradcam(file, write_path="data/gradcam_out/"):
    """
    return the GradCAM visualization overlaid on the input image file
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if type(file) is str:
        img = Image.open(file)
        try:
            img = preprocess(img)
        except AssertionError:
            print("Check the shape of the image, should be (28, 28, 1)")
    
    else:
        img = file
        try:
            img = preprocess(img)
        except AssertionError:
            print("Check the shape of the image, should be (28, 28, 1)")
    
    image = transform(img=img)

    # print(image.shape)

    model = SimpleMNISTClassifier(input_dim=(1, 28, 28), num_classes=10)
    
    grad_cam_model = GRADCAM_Model(model, saved_path="models/mnist_10_05-08-23 16-09.pt")
    grad_cam_model.eval()

    logits = grad_cam_model(image.unsqueeze(0))
    preds = logits.argmax(axis=-1)
    print("Predicted_class: {}".format(preds.numpy()[0]))

    # backward on the most probable output
    logits[:, preds[0]].backward()

    # get the gradients
    gradients = grad_cam_model.get_activation_gradients()

    # pool the gradients across channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations
    activations = grad_cam_model.get_activation(image.unsqueeze(0)).detach()

    # weigh the channels by the corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, ...] *= pooled_gradients[i]
    
    # get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on the heatmap
    # as mentioned in the paper
    heatmap = torch.relu(heatmap)

    # normalize 
    heatmap /= (torch.max(heatmap) + 1e-6)

    # overlay the heatmap on the image
    resized_heatmap = cv2.resize(heatmap.numpy(), [image.shape[1], image.shape[2]])
    resized_heatmap = resized_heatmap * 255
    resized_heatmap = resized_heatmap.astype("uint8")
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    # resized_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.cvtColor(np.array(image.squeeze()), cv2.COLOR_GRAY2RGB) * 255
    superimposed_img = resized_heatmap * 0.4 + image_rgb.astype("uint8")
    # superimposed_img = superimposed_img.astype('uint8')
    # print(image_rgb.max(), image_rgb.min())

    if write_path is not None:
        cv2.imwrite(os.path.join(write_path, "img_class_{}.png".format(preds[0].numpy())), img=superimposed_img)
    
    return superimposed_img[..., ::-1] / 255, heatmap, preds.numpy()[0]


if __name__ == "__main__":
    sup_img, heatmap, recognized_digit = gradcam("data/mnist_png/train/6/218.png")

    plt.matshow(heatmap)
    plt.show()
    plt.imshow(sup_img)
    plt.show()