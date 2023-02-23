import os
import math
import random
import shutil

SCALE_FACTOR = 1.0
NUM_CLASSES = 3
TILE_SHAPE = (257, 257)
TILE_HEIGHT, TILE_WIDTH = TILE_SHAPE

INPUT_DATASET_DIR = f"ssa_sidewalks_cityscapes10_{SCALE_FACTOR}_{TILE_WIDTH}x{TILE_HEIGHT}_tiles"
OUTPUT_DATASET_DIR = f"ssa_sidewalks_cityscapes10_{SCALE_FACTOR}_{TILE_WIDTH}x{TILE_HEIGHT}_tiles_final_dataset"

TRAIN_PERCENTAGE = 0.8

def main():
    if not os.path.exists(INPUT_DATASET_DIR):
        return

    image_names_per_class = [os.listdir(os.path.join(INPUT_DATASET_DIR, "images", str(id)))[:100] for id in range(NUM_CLASSES)]

    images_count_per_class = len(image_names_per_class[0])
    assert all(len(image_names_per_class[id]) == images_count_per_class for id in range(NUM_CLASSES))
    train_images_count_per_class = math.ceil(images_count_per_class*TRAIN_PERCENTAGE)

    train_images_per_class = [random.sample(image_names_per_class[id], train_images_count_per_class) for id in range(NUM_CLASSES)]
    test_images_per_class = [[] for _ in range(NUM_CLASSES)]
    for id in range(NUM_CLASSES):
        train_images_set = set(train_images_per_class[id])
        test_images_per_class[id] = [image_name for image_name in image_names_per_class[id] if image_name not in train_images_set]

    if os.path.exists(OUTPUT_DATASET_DIR):
        shutil.rmtree(OUTPUT_DATASET_DIR)
    os.mkdir(OUTPUT_DATASET_DIR)
    os.mkdir(os.path.join(OUTPUT_DATASET_DIR, "train"))
    train_images_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "images")
    train_masks_dir = os.path.join(OUTPUT_DATASET_DIR, "train", "masks")
    os.mkdir(train_images_dir)
    os.mkdir(train_masks_dir)
    os.mkdir(os.path.join(OUTPUT_DATASET_DIR, "test"))
    test_images_dir = os.path.join(OUTPUT_DATASET_DIR, "test", "images")
    test_masks_dir = os.path.join(OUTPUT_DATASET_DIR, "test", "masks")
    os.mkdir(test_images_dir)
    os.mkdir(test_masks_dir)

    for id in range(NUM_CLASSES):
        for image_name in train_images_per_class[id]:
            image_path = os.path.join(INPUT_DATASET_DIR, "images", str(id), image_name)
            mask_path = os.path.join(INPUT_DATASET_DIR, "masks", str(id), os.path.splitext(image_name)[0] + ".png")
            assert os.path.exists(mask_path)

            shutil.copy(image_path, train_images_dir)
            shutil.copy(mask_path, train_masks_dir)

        for image_name in test_images_per_class[id]:
            image_path = os.path.join(INPUT_DATASET_DIR, "images", str(id), image_name)
            mask_path = os.path.join(INPUT_DATASET_DIR, "masks", str(id), os.path.splitext(image_name)[0] + ".png")
            assert os.path.exists(mask_path)

            shutil.copy(image_path, test_images_dir)
            shutil.copy(mask_path, test_masks_dir)


if __name__ == "__main__":
    main()
