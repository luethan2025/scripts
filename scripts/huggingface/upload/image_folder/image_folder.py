import argparse
import tempfile
from pathlib import Path

from datasets import Dataset, Features, Image
from huggingface_hub import HfApi


def create_dataset(image_paths):
    return Dataset.from_dict(
        {
            "image": [
                {"bytes": Path(image_path).read_bytes(), "path": None}
                for image_path in image_paths
            ]
        },
        features=Features({"image": Image()}),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Simple command-line interface.")
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Image folder.",
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
    parser.add_argument(
        "--single_paruqet",
        action="store_true",
        help="Upload all images as data/dataset.parquet.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(
        str(path)
        for path in Path(args.image_folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        if args.single_paruqet:
            dataset = create_dataset(files)
            dataset.to_parquet(str(data_dir / "dataset.parquet"))
        else:
            num_shards = (len(files) + args.shard_size - 1) // args.shard_size
            padding = len(str(num_shards - 1))

            for i, start in enumerate(range(0, len(files), args.shard_size)):
                end = min(start + args.shard_size, len(files))
                shard = create_dataset(files[start:end])

                parquet_path = data_dir / f"dataset-{i:0{padding}d}.parquet"
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
