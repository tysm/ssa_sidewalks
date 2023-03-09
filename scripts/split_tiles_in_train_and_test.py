import os
import argparse
import math
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into train and test datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-percentage", type=float, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        return
    num_classes = len(os.listdir(os.path.join(args.dataset_dir, "images")))

    # get all image names per class
    image_names_per_class = [os.listdir(os.path.join(args.dataset_dir, "images", str(index))) for index in range(num_classes)]

    # shuffle image names per class
    for index in range(num_classes):
        random.shuffle(image_names_per_class[index])

    # get image count per class
    images_count_per_class = len(image_names_per_class[0])
    assert all(len(image_names_per_class[index]) == images_count_per_class for index in range(num_classes))

    # get number of train images per class
    train_images_count_per_class = math.ceil(images_count_per_class*args.train_percentage)

    # get train and test images per class
    train_images_per_class = [image_names_per_class[index][:train_images_count_per_class] for index in range(num_classes)]
    test_images_per_class = [image_names_per_class[index][train_images_count_per_class:] for index in range(num_classes)]

    if os.path.exists(os.path.join(args.output_dir, "train")):
        shutil.rmtree(os.path.join(args.output_dir, "train"))
    os.makedirs(os.path.join(args.output_dir, "train"))

    train_images_dir = os.path.join(args.output_dir, "train", "images")
    train_masks_dir = os.path.join(args.output_dir, "train", "masks")
    os.makedirs(train_images_dir)
    os.makedirs(train_masks_dir)

    if os.path.exists(os.path.join(args.output_dir, "test")):
        shutil.rmtree(os.path.join(args.output_dir, "test"))
    os.makedirs(os.path.join(args.output_dir, "test"))

    test_images_dir = os.path.join(args.output_dir, "test", "images")
    test_masks_dir = os.path.join(args.output_dir, "test", "masks")
    os.makedirs(test_images_dir)
    os.makedirs(test_masks_dir)

    # copy images, masks and class_colors.txt
    for index in range(num_classes):
        for image_name in train_images_per_class[index]:
            image_path = os.path.join(args.dataset_dir, "images", str(index), image_name)
            mask_path = os.path.join(args.dataset_dir, "masks", str(index), os.path.splitext(image_name)[0] + ".png")
            assert os.path.exists(mask_path)

            shutil.copy(image_path, train_images_dir)
            shutil.copy(mask_path, train_masks_dir)

        for image_name in test_images_per_class[index]:
            image_path = os.path.join(args.dataset_dir, "images", str(index), image_name)
            mask_path = os.path.join(args.dataset_dir, "masks", str(index), os.path.splitext(image_name)[0] + ".png")
            assert os.path.exists(mask_path)

            shutil.copy(image_path, test_images_dir)
            shutil.copy(mask_path, test_masks_dir)
    shutil.copy(os.path.join(args.dataset_dir, "class_colors.txt"), os.path.join(args.output_dir, "train", "class_colors.txt"))
    shutil.copy(os.path.join(args.dataset_dir, "class_colors.txt"), os.path.join(args.output_dir, "test", "class_colors.txt"))


if __name__ == "__main__":
    main()
