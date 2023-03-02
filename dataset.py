import os
import glob

import numpy as np
from PIL import Image
from torch.utils import data


class Dataset(data.Dataset): 
    def __init__(self, items, transforms=None, images_transforms=None, masks_transforms=None):
        super(Dataset, self).__init__()
        self.items = items
        self.transforms = transforms
        self.images_transforms = images_transforms
        self.masks_transforms = masks_transforms

    @staticmethod
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

    def read_item(self, index):
        assert index < len(self)

        image_path, mask_path = self.items[index]

        with Image.open(image_path).convert("RGB") as image:
            image_data = np.array(image)
        with Image.open(mask_path).convert("L") as mask:
            mask_data = np.array(mask)

        return image_data, mask_data

    def apply_transforms(self, item):
        image_data, mask_data = item

        if self.transforms is not None:
            results = self.transforms(image=image_data, mask=mask_data)
            image_data, mask_data = results["image"], results["mask"]

        if self.images_transforms is not None:
            image_data = self.images_transforms(image=image_data)["image"]

        if self.masks_transforms is not None:
            mask_data = self.masks_transforms(image=mask_data)["image"]

        return image_data, mask_data

    def __getitem__(self, index):
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_path: image file path, string
        - mask_path: mask file path, string
        """
        assert index < len(self)

        # Get image and mask arrays
        item_data = self.read_item(index)

        # Apply image and mask transforms
        item_data = self.apply_transforms(item_data)

        return item_data + self.items[index]

    def __len__(self):
        return len(self.items)
