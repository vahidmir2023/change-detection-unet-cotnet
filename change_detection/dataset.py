import os, tqdm, cv2
import torch
from pathlib import Path
from glob import glob
from torchvision.io import read_image
from torchvision import transforms as T


__all__ = [
    "LevirCDDataset",
    "make_cropped_dataset",
]


def make_cropped_dataset(ds_path, crop_size = (128, 128), stride = (64, 64), img_format="png"):
    for folder in ('A', 'B', 'label'):
        img_folder = str(ds_path / folder)
        cropping_folder_path = f"{ds_path}/{folder}_{crop_size[0]}_{crop_size[1]}"
        os.makedirs(cropping_folder_path, exist_ok=True)
        print(f"Working on {folder} images")
        for img_path in tqdm.tqdm(glob(img_folder + f"/*.{img_format}")):
            img = cv2.imread(img_path)
            org_width, org_height, _ = img.shape
            for x in range(0, org_width, stride[0]):
                cropped_width = x + crop_size[0]
                for y in range(0, org_height, stride[1]):
                    cropped_height = y + crop_size[1]
                    if cropped_width < org_width and cropped_height < org_height:
                        cropped_img = img[x:cropped_width, y:cropped_height, :]
                        basename = os.path.basename(img_path).split(".")[0]
                        cv2.imwrite(
                            f"{cropping_folder_path}/{basename}_{x}_{cropped_width}_{y}_{cropped_height}.{img_format}",
                            cropped_img
                        )


class LevirCDDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, partition = "train", crop_size: tuple = (128, 128), img_format = "png"):
        assert partition in ("train", "val", "test")
        self.imgs_path = {}
        self.len_imgs = -1
        for folder in ("A", "B", "label"):
            path = str(ds_path / partition / f"{folder}_{crop_size[0]}_{crop_size[1]}")
            self.imgs_path[folder] = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(f".{img_format}")]
            self.imgs_path[folder].sort()

            if self.len_imgs == -1:
                self.len_imgs = len(self.imgs_path[folder])
            else:
                assert self.len_imgs == len(self.imgs_path[folder])

    def __len__(self):
        return self.len_imgs
    
    def __getitem__(self, index):
        """
        Returns stacked A and B and the label image as well
        """
        img1 = read_image(self.imgs_path["A"][index])
        img2 = read_image(self.imgs_path["B"][index])
        img = torch.concat([img1, img2], axis=0) / 255.
        label = T.Grayscale()(read_image(self.imgs_path["label"][index])) / 255.
        label = torch.concat([1-label, label], axis=0)
        return img, label