import argparse
import json
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a NeRF synthetic dataset from the Hugging Face Hub."
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
        help="Local NeRF synthetic dataset output directory.",
    )
    return parser.parse_args()


def restore_split(repo_id, source_split, output_split, parquet_files, output_dir):
    local_parquet_files = [
        hf_hub_download(
            repo_id=repo_id,
            filename=parquet_file,
            repo_type="dataset",
        )
        for parquet_file in parquet_files
    ]
    dataset = load_dataset(
        "parquet",
        data_files={source_split: local_parquet_files},
    )[source_split]

    frames = []
    for index, example in enumerate(dataset):
        frame_path = f"{output_split}/{index + 1:04d}"
        image_path = output_dir / Path(frame_path).with_suffix(".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        example["image"].save(image_path, format="PNG")

        frame = {
            "file_path": frame_path,
            "transform_matrix": example["transform_matrix"],
        }
        if "rotation" in example:
            frame["rotation"] = example["rotation"]
        frames.append(frame)

    metadata = {
        "camera_angle_x": dataset[0]["camera_angle_x"],
        "frames": frames,
    }
    metadata_path = output_dir / f"transforms_{output_split}.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser() / args.repo_id.rsplit("/", 1)[-1]

    split_names = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    repository_files = HfApi().list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    parquet_files_by_split = {
        source_split: sorted(
            file_name
            for file_name in repository_files
            if len(PurePosixPath(file_name).parts) == 2
            and PurePosixPath(file_name).parts[0] == source_split
            and file_name.endswith(".parquet")
        )
        for source_split in split_names
    }
    available_splits = {
        source_split: parquet_files
        for source_split, parquet_files in parquet_files_by_split.items()
        if parquet_files
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for source_split, parquet_files in available_splits.items():
        restore_split(
            args.repo_id,
            source_split,
            split_names[source_split],
            parquet_files,
            output_dir,
        )


if __name__ == "__main__":
    main()