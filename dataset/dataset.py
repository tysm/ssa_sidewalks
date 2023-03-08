import numpy as np
from PIL import Image
from torch.utils import data


class Dataset(data.Dataset): 
    def __init__(self, items, class_colors, transforms=None, images_transforms=None, masks_transforms=None):
        super(Dataset, self).__init__()
        self.items = items
        self.class_colors = class_colors
        self.num_classes = len(class_colors)
        self.transforms = transforms
        self.images_transforms = images_transforms
        self.masks_transforms = masks_transforms

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
