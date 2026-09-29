import argparse
import tempfile
from pathlib import Path

from datasets import Dataset, Image
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple command-line interface."
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(
        str(path)
        for path in Path(args.image_folder).rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    dataset = Dataset.from_dict({"image": files})
    dataset = dataset.cast_column("image", Image())

    num_shards = (len(dataset) + args.shard_size - 1) // args.shard_size
    padding = len(str(num_shards - 1))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        for i, start in enumerate(
            range(0, len(dataset), args.shard_size)
        ):
            end = min(start + args.shard_size, len(dataset))
            shard = dataset.select(range(start, end))

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
