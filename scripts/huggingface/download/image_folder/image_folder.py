import argparse
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download an image-folder dataset from the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Local output directory for the downloaded images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser() / args.repo_id.rsplit("/", 1)[-1]

    repository_files = HfApi().list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    parquet_files = sorted(
        file_name
        for file_name in repository_files
        if len(PurePosixPath(file_name).parts) == 2
        and PurePosixPath(file_name).parts[0] == "data"
        and file_name.endswith(".parquet")
    )

    local_parquet_files = [
        hf_hub_download(
            repo_id=args.repo_id,
            filename=parquet_file,
            repo_type="dataset",
        )
        for parquet_file in parquet_files
    ]
    dataset = load_dataset(
        "parquet",
        data_files={"train": local_parquet_files},
    )["train"]

    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {"JPEG": ".jpg", "PNG": ".png"}
    for index, example in enumerate(dataset, start=1):
        image = example["image"]
        image_format = image.format
        if image_format in extensions:
            image.save(output_dir / f"{index:04d}{extensions[image_format]}")


if __name__ == "__main__":
    main()
