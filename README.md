# Basic Usage of CLIP-GradCAM

This  repo implements a GradCAM method based on the CLIP model. It is inspired by the official code: [CLIP-GradCAM Example](https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/clip_example.py).

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- `opencv-python`
- `numpy`
- `torch`
- `pytorch-grad-cam`

# How to Use

Adjust the parameters as needed:

- image-path: Specify the path to the input image.  
- labels: Provide a list of descriptions for the image.  
- prompt: Select one description from the list to use as the prompt.  
- method: Specify the method for generating the class activation map. Supported methods include gradcam, gradcam++, scorecam, and others.

# Example Command

Run the script with the following command:

```bash
python main.py --image-path "cat_and_dog.jpg" --labels ["a cat", "a dog"] --prompt "a cat" --method "gradcam"

