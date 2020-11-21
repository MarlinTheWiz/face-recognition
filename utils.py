"""
"""
import torch
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np


def _rotate_img(img, landmarks):
    """
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    if dy == 0:
        return img
    angle = 90 - np.degrees(np.arctan2(dy, dx))
    if left_eye[1] < right_eye[1]:
        direction = -1
    else:
        direction = 1
    
    if direction == -1:
        angle = 90 - angle
    return img.rotate(direction * angle)


def _crop_img(img, landmarks):
    """
    """
    intra_eye_dist = np.abs(landmarks[0][0] - landmarks[1][0]) // 2
    min_y = np.maximum(int(landmarks[0][1] - 1.3 * intra_eye_dist), 0)
    max_y = np.minimum(int(landmarks[0][1] + 3.2 * intra_eye_dist), img.size[1])
    min_x = np.maximum(int(landmarks[0][0] - 0.7 * intra_eye_dist), 0)
    max_x = np.minimum(int(landmarks[1][0] + 0.7 * intra_eye_dist), img.size[0])
    return img.crop((min_x, min_y, max_x, max_y))


def _stddev(img, mean_point, point, k_size):
    """
    """
    k_sizeX, k_sizeY = k_size // 2, k_size // 2

    y_start = point[1] - k_sizeY if 0 < point[1] - k_sizeY < img.shape[0] else 0
    y_end = point[1] + k_sizeY + 1 if 0 < point[1] + k_sizeY + 1 < img.shape[0] else img.shape[0] - 1

    x_start = point[0] - k_sizeX if 0 < point[0] - k_sizeX < img.shape[1] else 0
    x_end = point[0] + k_sizeX + 1 if 0 < point[0] + k_sizeX + 1 < img.shape[1] else img.shape[1] - 1

    patch = (img[y_start:y_end, x_start:x_end] - mean_point) ** 2
    total = np.sum(patch)
    n = patch.size

    return 1 if total == 0 or n == 0 else np.sqrt(total / float(n))


def _intensity_normalization(img, k_size=15):
    """
    """
    blur = cv2.GaussianBlur(img, (k_size, k_size), 0, 0).astype(np.float64)
    new_img = np.ones(img.shape, dtype=np.float32) * 127.

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            original = img[y, x]
            gauss = blur[y, x]
            desvio = _stddev(img, gauss, [x, y], k_size)

            novo_pixel = 127
            if desvio > 0:
                novo_pixel = (original - gauss) / float(desvio)

            new_val = np.clip((novo_pixel * 127 / float(2.0)) + 127, 0, 255)
            new_img[y, x] = new_val
    return new_img.astype('uint8')


def _get_landmarks(shape):
    """
    """
    parts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    
    left_eye = ((parts[36][0] + parts[39][0])//2, (parts[36][1] + parts[39][1])//2)
    right_eye = ((parts[42][0] + parts[45][0])//2, (parts[42][1] + parts[45][1])//2)
    nose = parts[33]
    left_mouth = parts[48]
    right_mouth = parts[54]
    landmarks = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype='float32')
    return landmarks


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, landmarks):
        for t in self.transforms:
            if isinstance(t, (Rotation, Crop)):
                image = t(image, landmarks)
            elif isinstance(t, ToTensor):
                image, target = t(image, target)
            else:
                image = t(image)
        return image, target


class Resize(object):
    def __init__(self, size, resample=Image.LINEAR):
        self.size = size
        self.resample = resample
    
    def __call__(self, img):
        return img.resize(self.size, resample=self.resample)


class Rotation(object):
    def __call__(self, img, landmarks):
        return _rotate_img(img, landmarks)


class Crop(object):
    def __call__(self, img, landmarks):
        return _crop_img(img, landmarks)


class IntensityNormalize(object):
    def __call__(self, img):
        img = np.array(img.convert('L'))
        return Image.fromarray(_intensity_normalization(img))


class SyntheticSample(object):
    def __call__(self, img):
        pass


class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), torch.tensor(label)