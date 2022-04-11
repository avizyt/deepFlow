from hashlib import new
import os, shutil, pathlib
from unicodedata import category

img_path = pathlib.Path("../dataset/facemask_images")

new_img_path = pathlib.Path("../dataset/facemask_images_small")

img_category = ["with_mask", "without_mask"]

def make_subset(subset_name, start_idx, end_idx, original_dir, new_base_dir, categories):
    for category in categories:
        new_dir = new_base_dir / subset_name / category
        if not new_dir.exists():
            os.makedirs(new_dir)
        fnames = [os.listdir(original_dir / category)[i] for i in range(start_idx, end_idx)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / category / fname, dst=new_dir/fname)


make_subset("train", 0, 450, img_path, new_img_path, img_category)
make_subset("val", 450, 550, img_path, new_img_path, img_category)
make_subset("test", 550, 680, img_path, new_img_path, img_category)