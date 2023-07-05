import os, tqdm, cv2
from pathlib import Path
from glob import glob

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
