import argparse
import shutil
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple local image folders into one."
    )
    parser.add_argument(
        "--dataset_dir",
        nargs="+",
        required=True,
        type=Path,
        help="Image dataset directories. The first directory is the output.",
    )
    return parser.parse_args()


def image_files(dataset_dir):
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def nonconflicting_path(output_dir, relative_path):
    destination = output_dir / relative_path
    suffix = 1
    while destination.exists():
        destination = output_dir / relative_path.with_name(
            f"{relative_path.stem}_{suffix}{relative_path.suffix}"
        )
        suffix += 1
    return destination


def combine_datasets(dataset_dirs):
    output_dir = dataset_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_images = image_files(output_dir)
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        staged_images = {}
        try:
            for index, image_path in enumerate(base_images):
                staged_path = Path(temp_dir) / str(index)
                shutil.move(str(image_path), str(staged_path))
                staged_images[image_path] = staged_path
        finally:
            for image_path, staged_path in staged_images.items():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged_path), str(image_path))

    for dataset_dir in dataset_dirs[1:]:
        for image_path in image_files(dataset_dir):
            relative_path = image_path.relative_to(dataset_dir)
            destination = nonconflicting_path(output_dir, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)


def main():
    args = parse_args()
    dataset_dirs = [path.expanduser() for path in args.dataset_dir]
    combine_datasets(dataset_dirs)


if __name__ == "__main__":
    main()