import argparse
import os
import zipfile
import shutil

import requests


def download_file_from_google_drive(file_url: str, download_path: str, replace_if_exists: bool = False) -> None:
    file_id = file_url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}&confirm=1"

    # Check if download_path exists
    if os.path.exists(download_path):
        if replace_if_exists:
            if os.path.isdir(download_path):
                shutil.rmtree(download_path)
            else:
                os.remove(download_path)
        else:
            raise ValueError(f"{download_path} already exists")

    # Download the file from the URL and save it to download_path
    with requests.get(download_url, stream=True) as response, open(download_path, 'wb') as out_file:
        # Raise an error if the response code is not successful
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)


def unzip_file(file_path: str, output_dir: str) -> None:
    # Check if output_dir exists and is a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError(f"{output_dir} is not a directory")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Extract the contents of the zip file and save it to the output directory
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Download and unzip a zip dataset from Google Drive.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to download')
    parser.add_argument('dataset_url', type=str, help='URL of the dataset to download')
    parser.add_argument('download_dir', type=str, help='Directory to download the dataset to')
    parser.add_argument('unzip_dir', type=str, help='Directory to extract the contents of the dataset to')
    args = parser.parse_args()

    # Ensure that download_dir exists and is a directory
    if os.path.exists(args.download_dir):
        if not os.path.isdir(args.download_dir):
            raise ValueError(f"{args.download_dir} is not a directory")
    else:
        os.makedirs(args.download_dir)

    # Ensure that unzip_dir exists and is a directory
    if os.path.exists(args.unzip_dir):
        if not os.path.isdir(args.unzip_dir):
            raise ValueError(f"{args.unzip_dir} is not a directory")
    else:
        os.makedirs(args.unzip_dir)

    download_path = os.path.join(args.download_dir, f"{args.dataset_name}.zip")

    # Download the zip file to download_path
    download_file_from_google_drive(args.dataset_url, download_path, replace_if_exists=True)

    # Extract the contents of the zip file to unzip_dir
    unzip_file(download_path, os.path.join(args.unzip_dir, args.dataset_name))

if __name__ == '__main__':
    main()
