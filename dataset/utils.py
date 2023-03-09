import os
import glob

import torch
from torch.utils import data

from dataset.dataset import Dataset


def find_items(images_dir, masks_dir, image_ext, mask_ext):
    """
    Find image and segmentation mask files and return a list of
    tuples of them.
    """
    items = []
    images = glob.glob(f"{images_dir}/**/*.{image_ext}", recursive=True)
    for image_path in images:
        _images_dir, image_basename = os.path.split(image_path)

        common_path = ""
        while not os.path.samefile(_images_dir, images_dir):
            common_path = os.path.join(os.path.basename(_images_dir))
            _images_dir = os.path.dirname(_images_dir)

        mask_basename = f"{os.path.splitext(image_basename)[0]}.{mask_ext}"
        mask_path = os.path.join(masks_dir, common_path, mask_basename)
        assert os.path.exists(mask_path)

        items.append((image_path, mask_path))
    return items


def read_class_colors(dataset_dir):
    """
    Reads the class_colors.txt dataset config and return a color list.
    The list should map each index to a rgb color and assume unlabeled data
    as 0 index.
    """
    with open(os.path.join(dataset_dir, "class_colors.txt"), "r") as colors_config:
        colors = list(list(int(value) for value in line.split()[:3]) for line in colors_config.readlines())
    return [[0, 0, 0], *colors]


def setup_loader(dataset_dir, transforms, images_transforms, masks_transforms, batch_size, workers, shuffle, drop_last):
    items = find_items(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "masks"), "jpg", "png")
    class_colors = read_class_colors(dataset_dir)
    dataset = Dataset(items, class_colors, transforms, images_transforms, masks_transforms)
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle, drop_last=drop_last)


def compute_median_frequency_class_balancing_weights(loader, device):
    with torch.no_grad():
        num_classes = loader.dataset.num_classes
        accumulator = torch.zeros(num_classes).to(device=device)
        for _, masks, _, _ in loader:
            masks = masks.long().to(device=device)

            # create one-hot encoding of masks for each class
            masks_one_hot = torch.zeros((masks.size(0), num_classes, masks.size(2), masks.size(3))).to(device=device)
            masks_one_hot.scatter_(1, masks, 1)

            accumulator += masks_one_hot.sum(dim=(0, 2, 3))
        return accumulator.median()/(accumulator + 1e-6) # adding epsilon to avoid divide by zero errors
