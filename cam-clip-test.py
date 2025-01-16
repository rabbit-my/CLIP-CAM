"""基于clip模型，可以根据不同的提示prompts生成不同的grad cam"""

import argparse
import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("The transformers package is not installed. Please install it to use CLIP.")
    exit(1)

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Torch device to use')
    parser.add_argument('--image-path', type=str, default='/home/codebase/Yinmi/medical-lora/DATA/tiff_xiangya_frame_update/images/cyst/M0008_2021_P0001233_circle_2.0x2.5_C8_S8_0.png', help='Input image path')
    parser.add_argument('--labels', type=str, nargs='+', default=["a medical photo","A black-and-white medical image characterized by an extruded gray-white-gray hierarchy in the middle area","black area", "hierarchical structure", "convex curve", "a elliptical", "middle and lower parts"],
                        help='Recognition labels')
    parser.add_argument('--prompt', type=str, default="a elliptical", help='The target object to generate heatmap for')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam', help='Method: gradcam, gradcam++, scorecam, etc.')

    args = parser.parse_args()
    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class ImageClassifier(nn.Module):
    def __init__(self, labels):
        super(ImageClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained("/home/user/.cache/huggingface/hub/models--openai--clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained(
            "/home/user/.cache/huggingface/hub/models--openai--clip-vit-base-patch16")
        self.labels = labels

    def forward(self, x):
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)
        outputs = self.clip(pixel_values=x, input_ids=text_inputs['input_ids'].to(self.clip.device),
                            attention_mask=text_inputs['attention_mask'].to(self.clip.device))

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        for label, prob in zip(self.labels, probs[0]):
            print(f"{label}: {prob:.4f}")
        return probs


if __name__ == '__main__':
    args = get_args()

    methods = {"gradcam": GradCAM, "scorecam": ScoreCAM, "gradcam++": GradCAMPlusPlus, "ablationcam": AblationCAM,
               "xgradcam": XGradCAM, "eigencam": EigenCAM, "eigengradcam": EigenGradCAM, "layercam": LayerCAM,
               "fullgrad": FullGrad}

    if args.method not in methods:
        raise Exception(f"Method should be one of {list(methods.keys())}")

    labels = args.labels
    model = ImageClassifier(labels).to(torch.device(args.device)).eval()

    target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(args.device)

    # Dynamic target selection based on the prompt
    try:
        target_index = labels.index(args.prompt)  # Find the index of the prompt in the label list
        print(target_index)
        targets = [ClassifierOutputTarget(target_index)]
    except ValueError:
        print(f"Prompt '{args.prompt}' not found in labels list. Using the highest probability target.")
        targets = None

    if args.method == "ablationcam":
        cam = methods[args.method](model=model, target_layers=target_layers, reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    grayscale_cam = grayscale_cam[0, :]  # Take the first (and only) item in the batch
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # Save the result to a file
    output_filename = f'{args.prompt}_{args.method}_cam.jpg'
    cv2.imwrite(output_filename, cam_image)
    print(f"Heatmap saved to {output_filename}")

"""run:
    python cam-clip-test.py --image-path "cat_and_dog.jpg" --prompt "a cat" --method "gradcam"
"""