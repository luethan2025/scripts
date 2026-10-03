import argparse
import json
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a NeRF synthetic dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="NeRF synthetic dataset root containing transforms_*.json and split folders.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        required=True,
        help="Commit message.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10_000,
        help="Maximum number of images per shard.",
    )
    return parser.parse_args()


def load_split(dataset_dir, source_split):
    metadata_path = dataset_dir / f"transforms_{source_split}.json"
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    examples = []
    for frame in metadata["frames"]:
        frame_path = Path(frame["file_path"])
        image_path = dataset_dir / (
            frame_path if frame_path.suffix else frame_path.with_suffix(".png")
        )

        examples.append(
            {
                "image": str(image_path),
                "camera_angle_x": metadata["camera_angle_x"],
                "rotation": frame["rotation"],
                "transform_matrix": frame["transform_matrix"],
            }
        )

    return Dataset.from_list(examples).cast_column("image", Image())


def main():
    args = parse_args()    

    dataset_dir = Path(args.dataset_dir).expanduser()

    splits = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    dataset = DatasetDict(
        {
            split: load_split(dataset_dir, source_split)
            for split, source_split in splits.items()
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        for split, split_dataset in dataset.items():
            num_shards = (len(split_dataset) + args.shard_size - 1) // args.shard_size
            padding = len(str(num_shards - 1))
            split_data_dir = data_dir / split
            split_data_dir.mkdir()

            for i, start in enumerate(
                range(0, len(split_dataset), args.shard_size)
            ):
                end = min(start + args.shard_size, len(split_dataset))
                shard = split_dataset.select(range(start, end))
                parquet_path = split_data_dir / f"dataset-{i:0{padding}d}.parquet"
                shard.to_parquet(str(parquet_path))

        api = HfApi()

        api.upload_folder(
            folder_path=str(temp_dir),
            path_in_repo="",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()
