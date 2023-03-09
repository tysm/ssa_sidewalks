import argparse
import os
import shutil
import random
import uuid

import numpy as np
from PIL import Image


IMAGES_DIR = "imgsFine/leftImg8bit/default"
IMAGES_SUFFIX = "_leftImg8bit.JPG"

MASKS_DIR = "gtFine/default"
MASKS_SUFFIX = "_gtFine_labelIds.png"


def create_tiles_dataset(base_dataset_dir: str, num_classes: int, scale_factor: float, max_num_tiles_per_image_per_class: int, tile_height: int, tile_width: int, output_dataset_dir: str) -> None:
    # Check that dataset_dir exists and is a directory
    if os.path.exists(base_dataset_dir):
        if not os.path.isdir(base_dataset_dir):
            raise ValueError(f"{base_dataset_dir} is not a directory")
    else:
        raise ValueError(f"{base_dataset_dir} does not exist")

    base_images_dir = os.path.join(base_dataset_dir, IMAGES_DIR)
    base_masks_dir = os.path.join(base_dataset_dir, MASKS_DIR)

    # Ensure base_dataset_dir contains base_images_dir directory
    if os.path.exists(base_images_dir):
        if not os.path.isdir(base_images_dir):
            raise ValueError(f"{base_images_dir} is not a directory")
    else:
        raise ValueError(f"{base_images_dir} does not exist")

    # Ensure base_dataset_dir contains base_masks_dir directory
    if os.path.exists(base_masks_dir):
        if not os.path.isdir(base_masks_dir):
            raise ValueError(f"{base_masks_dir} is not a directory")
    else:
        raise ValueError(f"{base_masks_dir} does not exist")

    # Check tile_height is odd
    if tile_height%2 != 1:
        raise ValueError(f"Tile height must be odd")

    # Check tile_width is odd
    if tile_width%2 != 1:
        raise ValueError(f"Tile width must be odd")

    # Ensure output_dataset_dir only has the following directories and that they're empty:
    # - ./images/[0..{num_classes})
    # - ./masks/[0..{num_classes})
    if os.path.exists(output_dataset_dir):
        if os.path.isdir(output_dataset_dir):
            shutil.rmtree(output_dataset_dir)
        else:
            os.remove(output_dataset_dir)
    os.makedirs(output_dataset_dir)

    output_images_dir = os.path.join(output_dataset_dir, "images")
    output_masks_dir = os.path.join(output_dataset_dir, "masks")
    os.mkdir(output_images_dir)
    os.mkdir(output_masks_dir)

    for id in range(num_classes):
        os.mkdir(os.path.join(output_images_dir, str(id)))
        os.mkdir(os.path.join(output_masks_dir, str(id)))

    # Crop random tiles from base images and masks, and store them in output_dataset_dir
    processed_count = 0
    half_tile_height = tile_height//2
    half_tile_width = tile_width//2
    base_image_names = os.listdir(base_images_dir)
    assert len(base_image_names) >= 1
    with Image.open(os.path.join(base_images_dir, base_image_names[0])) as image:
        data_height, data_width, _ = np.array(image).shape
    for image_name in base_image_names:
        image_id = image_name[:-len(IMAGES_SUFFIX)]
        print(f"processing: {image_id}")

        # Get mask as numpy array
        mask_path = os.path.join(base_masks_dir, image_id + MASKS_SUFFIX)
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask, dtype=np.uint8)
        else:
            mask_array = np.zeros((data_height, data_width), dtype=np.uint8)

        # Select up to max_num_tiles_per_image_per_class random class pixels to be the center of to be extracted tiles
        indexes_per_class = [[] for _ in range(num_classes)]
        for i in range(half_tile_height, data_height-half_tile_height):
            for j in range(half_tile_width, data_width-half_tile_width):
                # Check mask_array[i][j] is a valid class index
                if mask_array[i][j] >= num_classes:
                    raise ValueError(f"{mask_array[i][j]} is an invalid class")

                indexes_per_class[mask_array[i][j]].append((i, j))
        tiles_per_class = list(random.sample(indexes, min(len(indexes), max_num_tiles_per_image_per_class)) if indexes else [] for indexes in indexes_per_class)
        print(list(len(i) for i in tiles_per_class))

        # Crop images and their masks
        with Image.open(os.path.join(base_images_dir, image_name)) as image:
            for id in range(num_classes):
                for i, j in tiles_per_class[id]:
                    tile_uuid = str(uuid.uuid4())

                    image_tile = image.crop((j-half_tile_width, i-half_tile_height, j+half_tile_width+1, i+half_tile_height+1))
                    image_tile.save(os.path.join(output_images_dir, str(id), tile_uuid + ".jpg"), format="JPEG")
                    image_tile.close()

                    mask_tile = Image.fromarray(mask_array[i-half_tile_height:i+half_tile_height+1, j-half_tile_width:j+half_tile_width+1])
                    mask_tile.save(os.path.join(output_masks_dir, str(id), tile_uuid + ".png"), format="PNG")
                    mask_tile.close()
        processed_count += 1
        print(f"{processed_count}/{len(base_image_names)} processed")


def remove_extra_tiles_per_class(dataset_dir: str, num_classes: int) -> None:
    # Ensure dataset_dir only has the following directories:
    # - ./images/[0..{num_classes})
    # - ./masks/[0..{num_classes})

    # Ensure dataset_dir is a directory
    if os.path.exists(dataset_dir):
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not a directory")
    else:
        raise ValueError(f"{dataset_dir} does not exist")

    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    # Ensure images_dir is a directory
    if os.path.exists(images_dir):
        if not os.path.isdir(images_dir):
            raise ValueError(f"{images_dir} is not a directory")
    else:
        raise ValueError(f"{images_dir} does not exist")

    # Ensure masks_dir is a directory
    if os.path.exists(masks_dir):
        if not os.path.isdir(masks_dir):
            raise ValueError(f"{masks_dir} is not a directory")
    else:
        raise ValueError(f"{masks_dir} does not exist")

    # Ensure images_dir/id and masks_dir/id are directories
    for id in range(num_classes):
        cur_images_dir = os.path.join(images_dir, str(id))
        if os.path.exists(cur_images_dir):
            if not os.path.isdir(cur_images_dir):
                raise ValueError(f"{cur_images_dir} is not a directory")
        else:
            raise ValueError(f"{cur_images_dir} does not exist")

        cur_masks_dir = os.path.join(masks_dir, str(id))
        if os.path.exists(cur_masks_dir):
            if not os.path.isdir(cur_masks_dir):
                raise ValueError(f"{cur_masks_dir} is not a directory")
        else:
            raise ValueError(f"{cur_masks_dir} does not exist")

    image_names_per_class = [os.listdir(os.path.join(images_dir, str(id))) for id in range(num_classes)]
    mask_names_per_class = [os.listdir(os.path.join(masks_dir, str(id))) for id in range(num_classes)]

    # Ensure each class has the respective images and masks
    for id in range(num_classes):
        assert len(image_names_per_class[id]) == len(mask_names_per_class[id])

        mask_names_set = set(mask_names_per_class[id])
        for image_name in image_names_per_class[id]:
            mask_stem, _ = os.path.splitext(image_name)
            assert (mask_stem + ".png") in mask_names_set

    min_count = min(len(names) for names in image_names_per_class)
    print(min_count, "min image count")
    for id in range(num_classes):
        cur_count = len(image_names_per_class[id])
        if cur_count == min_count:
            continue

        deleting_images_names = random.sample(image_names_per_class[id], cur_count-min_count)
        print("deleting", len(deleting_images_names), "from class", id)
        for image_name in deleting_images_names:
            image_path = os.path.join(images_dir, str(id), image_name)
            assert os.path.exists(image_path)

            mask_stem, _ = os.path.splitext(image_name)
            mask_path = os.path.join(masks_dir, str(id), mask_stem + ".png")
            assert os.path.exists(mask_path)

            os.remove(image_path)
            os.remove(mask_path)


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Create a dataset by extracting random tiles separated by classes from a Cityscapes1.0-like dataset.')
    parser.add_argument('dataset_dir', type=str, help='Cityscapes1.0-like base dataset directory')
    parser.add_argument('scale_factor', type=float, help="Scale factor to apply on dataset images and masks before tile extraction")
    parser.add_argument('max_num_tiles_per_image_per_class', type=int, help='Maximum number of extracted tiles per class per image')
    parser.add_argument('tile_width', type=int, help='Width of to be extracted tiles')
    parser.add_argument('tile_height', type=int, help='Height of to be extracted tiles')
    parser.add_argument('output_dir', type=str, help='Directory to store the extracted tiles from the base dataset')
    args = parser.parse_args()

    # Check that dataset_dir exists and is a directory
    if os.path.exists(args.dataset_dir):
        if not os.path.isdir(args.dataset_dir):
            raise ValueError(f"{args.dataset_dir} is not a directory")
    else:
        raise ValueError(f"{args.dataset_dir} does not exist")
    with open(os.path.join(args.dataset_dir, "label_colors.txt"), "r") as f:
        num_classes = len(f.readlines()) + 1

    base_dataset_name = os.path.basename(args.dataset_dir)
    output_dataset_dir = os.path.join(args.output_dir, f"{base_dataset_name}_{args.scale_factor}_{args.tile_width}x{args.tile_height}_tiles")

    create_tiles_dataset(args.dataset_dir, num_classes, args.scale_factor, args.max_num_tiles_per_image_per_class, args.tile_height, args.tile_width, output_dataset_dir)
    remove_extra_tiles_per_class(output_dataset_dir, num_classes)

if __name__ == "__main__":
    main()
