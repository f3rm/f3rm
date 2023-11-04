import argparse
import hashlib
import os
import zipfile
from typing import NamedTuple

import gdown


class _Dataset(NamedTuple):
    name: str
    url: str
    md5: str


_datasets = [
    _Dataset(
        name="panda",
        url="https://drive.google.com/uc?id=15iNJo57bIM2NMyKVs4JU_nzNvFzJ1ZRU",
        md5="1aba91f804d02f5e9e886c189e58a8e0",
    ),
    _Dataset(
        name="panda_demos",
        url="https://drive.google.com/uc?id=1ljKdj5Jpkq3p5rJPMRZHfAVw5IKZKAJ5",
        md5="269034d94d6f1c34fd026c659d3a3bc7",
    ),
    _Dataset(
        name="rooms",
        url="https://drive.google.com/uc?id=1Kl84WHBN5VGTyuzKE9nd_HNrBQusEq21",
        md5="84dc1ec745c303a8e4da1c4a0871455e",
    ),
]

_name_to_dataset = {d.name: d for d in _datasets}
assert len(_name_to_dataset) == len(_datasets), "Dataset names must be unique!"


def get_dataset(name: str) -> _Dataset:
    if name not in _name_to_dataset:
        raise ValueError(f"Dataset {name} not found!")
    return _name_to_dataset[name]


def ask_yes_no(message: str) -> bool:
    while True:
        response = input(f"{message} [y/n]: ")
        if response == "y":
            return True
        elif response == "n":
            return False


def download_dataset(name: str, save_dir: str):
    print(f"=== Downloading {name} dataset ===")
    os.makedirs(save_dir, exist_ok=True)

    dataset = get_dataset(name)
    assert dataset.url.startswith("https://drive.google.com")

    zip_path = os.path.join(save_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        print(f"{zip_path} already exists! Delete it if you want to re-download.")
        return

    # If dataset directory already exists, ask user if they want to overwrite it before continuing
    dataset_dir = os.path.join(save_dir, name)
    if os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} already exists!")
        if ask_yes_no("Do you want to overwrite it?") is False:
            print(f"Skipping {name} dataset.")
            return

    # Download zip from Google Drive
    print(f"Downloading {name} datasets to {save_dir}...")
    gdown.download(dataset.url, output=zip_path)

    # Check md5 matches
    with open(zip_path, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    if actual_md5 != dataset.md5:
        raise RuntimeError(f"MD5 mismatch for {zip_path}! Expected {dataset.md5}, got {actual_md5}.")

    # Unzip and delete the zip file
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(save_dir)
    print(f"Successfully downloaded {name} dataset to {dataset_dir}.")
    os.remove(zip_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download F3RM datasets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "type",
        type=str,
        help="Type of dataset to download. Use 'all' to download all datasets.",
        nargs="?",
        choices=["all"] + list(_name_to_dataset.keys()),
        default="all",
    )
    parser.add_argument(
        "--save_dir", "-s", type=str, help="Directory to save the downloaded datasets to.", default="./datasets/f3rm"
    )
    args = parser.parse_args()

    # Download the datasets
    if args.type == "all":
        print(f"Downloading all {len(_datasets)} datasets...")
        for name in _name_to_dataset.keys():
            download_dataset(name, args.save_dir)
    else:
        download_dataset(args.type, args.save_dir)


if __name__ == "__main__":
    main()
