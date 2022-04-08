import os, shutil, pathlib

original_dir = "../dataset/intel_images"
new_base_dir = "../dataset/intel_images_subset"

def make_subset(subset_name, start_idx, end_idx):
    for category in("buildings", "forest", "glacier", "mountain", "sea", "street"):
        new_dir = new_base_dir / subset_name / category
        if not new_dir.exists():
            os.makedirs(new_dir)
        fnames = [os.listdir(original_dir / category)[i] for i in range(start_idx, end_idx)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / category/ fname, dst=new_dir / fname)

