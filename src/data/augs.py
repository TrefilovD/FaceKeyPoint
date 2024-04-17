import albumentations as A
import cv2
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw


TRANSFORMS = {
    "resize": A.Resize,
    # "normalize": A.Normalize
}


def get_transforms(input_size, mode):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(*input_size, cv2.INTER_NEAREST, True),
                A.OneOf([
                    A.Rotate(60, always_apply=True),
                    # A.VerticalFlip(always_apply=True),
                    A.HorizontalFlip(always_apply=True),
                ]),
                A.OneOf([
                    A.RandomBrightnessContrast(always_apply=True),
                    A.ShiftScaleRotate(always_apply=True),
                    A.GaussNoise(always_apply=True)
                ])
            ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=["class_labels"], remove_invisible=False),
        )
    else:
        return A.Compose(
            [
                A.Resize(*input_size, cv2.INTER_NEAREST, True),
            ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=["class_labels"], remove_invisible=False),
        )

KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=6, path="tmp.png", return_image=False):
    r = diameter / 2
    image = Image.fromarray(image.copy())
    image_d = ImageDraw.Draw(image)

    for (x, y) in keypoints:
        image_d.ellipse((x-r, y-r, x+r, y+r), fill=color)
    if path:
        image.save(path)
    if return_image:
        return image
