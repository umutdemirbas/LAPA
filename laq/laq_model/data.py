from PIL import Image

import os
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()

        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.folder}")

        self.folder_list = [
            entry for entry in sorted(self.folder.iterdir())
            if entry.is_dir() or entry.suffix.lower() == ".webm"
        ]
        if not self.folder_list:
            raise ValueError(f"No videos or frame folders found in: {self.folder}")

        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list)

    def _read_video_frame(self, video_path, frame_index):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def __getitem__(self, index):
        try :
            offset = self.offset

            item = self.folder_list[index]

            if item.is_dir():
                img_list = sorted(os.listdir(item), key=lambda x: int(x.split('.')[0][4:]))
                first_frame_idx = random.randint(0, len(img_list) - 1)
                first_frame_idx = min(first_frame_idx, len(img_list) - 1)
                second_frame_idx = min(first_frame_idx + offset, len(img_list) - 1)

                first_path = item / img_list[first_frame_idx]
                second_path = item / img_list[second_frame_idx]

                img = Image.open(first_path)
                next_img = Image.open(second_path)
            else:
                cap = cv2.VideoCapture(str(item))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {item}")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if frame_count <= 0:
                    raise RuntimeError(f"Could not determine frame count for {item}")

                first_frame_idx = random.randint(0, frame_count - 1)
                second_frame_idx = min(first_frame_idx + offset, frame_count - 1)

                img = self._read_video_frame(item, first_frame_idx)
                next_img = self._read_video_frame(item, second_frame_idx)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
