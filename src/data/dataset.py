import glob
import os.path as osp
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm

from data import get_transforms
from modules.face_detector import FaceDetector



class FaceDataset(torch.utils.data.Dataset):
    def __init__ (self, ds_dir, input_size, transforms=None, use_dlib_detector=False, use_mask=False, suffix = "jpg"):
        self.ds_dir = ds_dir
        self.input_size = input_size # w, h
        self.img_files = sorted(glob.glob(osp.join(ds_dir, "*" + suffix)))
        # self.markups_files = sorted(glob.glob(osp.join(ds_dir, "*pts")))
        self.markups_files = [".".join(p.split(".")[:-1]) + ".pts" for p in self.img_files]
        self.indexes = self.validation()
        self.transforms = transforms
        self.dlib_detector = FaceDetector() if use_dlib_detector else None
        self.use_mask = use_mask

        self.class_labels = list(range(68))
        self.eyes_idx=[36,39,42,45]

    def __len__(self) -> int:
        return len(self.indexes)

    def validation(self):
        """Оставляем разметку с 68 точками
        """
        idxs = []

        for idx, markup in enumerate(tqdm(self.markups_files, "validation dataset", ncols=100)):
            n_pts = np.loadtxt(markup, comments=("version:", "n_points:", "{", "}")).shape[0]
            if n_pts == 68:
                idxs += [idx]
        print(f"skipped samples {len(self.markups_files) - len(idxs)}")
        return idxs


    def read_data(self, idx):
        idx = self.indexes[idx]
        markup = np.loadtxt(self.markups_files[idx], comments=("version:", "n_points:", "{", "}"))
        img = np.array(Image.open(self.img_files[idx]).convert("RGB"))
        flg = False
        if self.dlib_detector:
            dets = self.dlib_detector(img)
            for det in dets:
                xmin, ymin, xmax, ymax = det.left(),det.top(),det.right(),det.bottom()
                if all(
                    [markup[i][0] > xmin for i in self.eyes_idx] +
                    [markup[i][1] > ymin for i in self.eyes_idx] +
                    [markup[i][0] < xmax for i in self.eyes_idx] +
                    [markup[i][1] < ymax for i in self.eyes_idx]
                ):
                    flg = True
                    break
        if self.dlib_detector is None or not flg:
            slice_pix = [0, 8, -8]
            xmin, ymin = np.min(markup, 0)
            xmax, ymax = np.max(markup, 0)
            xmin += slice_pix[random.randint(0, 2)]
            ymin += slice_pix[random.randint(0, 2)]
            xmax += slice_pix[random.randint(0, 2)]
            ymax += slice_pix[random.randint(0, 2)]

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1]-1, xmax)
        ymax = min(img.shape[0]-1, ymax)

        bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        offset = np.array(bbox[:2])

        h, w = img.shape[:2]

        t_img = A.crop(img, *bbox)
        t_markup =  markup - offset

        h, w = t_img.shape[:2]

        return {
            "image": t_img.astype(np.float32) / 255,
            "keypoints": t_markup.astype(np.float32),
            "class_labels": self.class_labels,
        }, offset

    def __getitem__(self, idx):
        data, offset = self.read_data(idx)
        if self.transforms:
            data = self.transforms(**data)

        data["keypoints"] = torch.tensor(data["keypoints"], dtype=torch.float32)

        horizontal_flipped = data["keypoints"][36, 0] > data["keypoints"][42, 0]
        if horizontal_flipped:
            data["keypoints"][:, 0] = self.input_size[0] - data["keypoints"][:, 0]

        vertical_flipped = data["keypoints"][8, 1] < data["keypoints"][27, 1]
        if vertical_flipped:
            data["keypoints"][:, 1] = self.input_size[1] - data["keypoints"][:, 1]


        if self.use_mask:
            mask = torch.stack([
                torch.BoolTensor(data["keypoints"][:, 0] >= 0) & \
                torch.BoolTensor(data["keypoints"][:, 0] < data["image"].shape[1]),
                torch.BoolTensor(data["keypoints"][:, 1] >= 0) & \
                torch.BoolTensor(data["keypoints"][:, 1] < data["image"].shape[0])
            ], 1)
        else:
            mask = torch.ones_like(data["keypoints"]).to(torch.bool)

        return (
            torch.tensor(data["image"], dtype=torch.float32).permute(2, 0, 1),
            data["keypoints"],
            torch.tensor(data["class_labels"], dtype=torch.int32).unsqueeze(-1).repeat((1, 2)).view(-1).to(torch.int64),
            torch.from_numpy(offset),
            mask.view(-1)
        )


def build_dataset(mode, cfg):
    transforms = get_transforms(mode=mode, **cfg.get("transforms"))
    return FaceDataset(transforms=transforms, **cfg.get("dataset"))



if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/e/WORK_DL/VisionLab/src")
    from data import get_transforms, vis_keypoints

    transforms = get_transforms((256, 256))
    ds = FaceDataset("/mnt/e/WORK_DL/datasets/landmarks_task/Menpo/test", transforms)
    for i in range(5):
        img, keypoints, _ = ds[i]
        vis_keypoints((img * 255).astype(np.uint8), keypoints.tolist(), diameter=2, path=f"tmp{i}.png")
        print(img.shape, keypoints.shape)