"""
"""
import glob
import os

import dlib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import utils

CLASS_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
SHAPE_DETECTOR_PATH = "shape_predictor_68_face_landmarks.dat"


class CKPlusDataset(Dataset):
    """
    """
    def __init__(self, data_dir, training=True, transform=ToTensor()):
        self.class_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.paths = glob.glob(os.path.join(data_dir, '*', '*'))
        self.labels = []
        for path in self.paths:
            class_name = os.path.split(os.path.dirname(path))[-1]
            self.labels.append(self.class_map[class_name])
        
        self.transform = transform
        if training:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(SHAPE_DETECTOR_PATH)
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        label = self.labels[index]
        gray = np.array(img.convert('L'))
        det = self.detector(gray, 1)
        shape = self.predictor(gray, det[0])
        landmarks = utils._get_landmarks(shape)
        if self.transform:
            img, label = self.transform(img, label, landmarks)
        return img, label


def load_data(data_dir, num_workers=4, batch_size=32, **kwargs):
    dataset = CKPlusDataset(data_dir, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)