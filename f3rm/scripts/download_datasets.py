import argparse
import hashlib
import os
import zipfile

import gdown

# Note: these are all ZIP files on Google Drive
_dataset_to_url = {
    "panda": "https://drive.google.com/uc?id=15iNJo57bIM2NMyKVs4JU_nzNvFzJ1ZRU",
    "rooms": "https://drive.google.com/uc?id=1Kl84WHBN5VGTyuzKE9nd_HNrBQusEq21",
}

_dataset_to_md5 = {
    "panda": "37b3b5240a1e68fec6cc13ac0a8dd644",
    "rooms": "c1606d4e0bfc9b5f702717fe209cd904",
}


def download_dataset(name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    url = _dataset_to_url[name]
    assert url.startswith("https://drive.google.com")

    zip_path = os.path.join(save_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        raise RuntimeError(f"{zip_path} already exists! Delete it if you want to re-download.")

    # Download from Google Drive
    print(f"Downloading {name} datasets to {save_dir}...")
    gdown.download(url, output=zip_path)

    # Check md5 matches
    expected_md5 = _dataset_to_md5[name]
    with open(zip_path, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {zip_path}! Expected {expected_md5}, got {actual_md5}.")

    # Unzip and delete the zip file
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(save_dir)
    os.remove(zip_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        type=str,
        help="Type of dataset to download. Use 'all' to download all datasets.",
        nargs="?",
        choices=["all"] + list(_dataset_to_url.keys()),
        default="all",
    )
    parser.add_argument(
        "--save_dir", "-s", type=str, help="Directory to save the downloaded datasets to.", default="./datasets"
    )
    args = parser.parse_args()

    # Download the datasets
    if args.type == "all":
        for name in _dataset_to_url.keys():
            download_dataset(name, args.save_dir)
    else:
        download_dataset(args.type, args.save_dir)


if __name__ == "__main__":
    main()
