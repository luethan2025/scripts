import argparse
import json
import shutil
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple NeRF synthetic datasets into one."
    )
    parser.add_argument(
        "--dataset_dir",
        nargs="+",
        required=True,
        type=Path,
        help="Dataset directories containing transforms_*.json and their images.",
    )
    return parser.parse_args()


def frame_number(frame_path):
    stem = frame_path.stem
    prefix = stem.rstrip("0123456789")
    suffix = stem[len(prefix) :]
    return int(suffix) if suffix else None


def renumber_frame_path(frame_path, frame_number, padding):
    stem = frame_path.stem
    prefix = stem.rstrip("0123456789")
    if prefix == stem:
        prefix = f"{stem}_"

    return frame_path.with_name(
        f"{prefix}{frame_number:0{padding}d}{frame_path.suffix}"
    )


def combine_split(dataset_dirs, output_dir, split):
    combined_metadata = None
    combined_frames = []
    sources = []
    base_dataset_index = None

    for dataset_index, dataset_dir in enumerate(dataset_dirs):
        metadata_path = dataset_dir / f"transforms_{split}.json"
        if not metadata_path.is_file():
            continue

        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        if combined_metadata is None:
            combined_metadata = dict(metadata)
            base_dataset_index = dataset_index

        sources.append((dataset_index, dataset_dir, metadata))

    if combined_metadata is None:
        return

    base_numbers = [
        number
        for frame in combined_metadata["frames"]
        if (number := frame_number(Path(frame["file_path"]))) is not None
    ]
    next_frame_number = max(base_numbers, default=0) + 1
    entries = []
    padding = 1

    for dataset_index, dataset_dir, metadata in sources:
        for frame in metadata["frames"]:
            frame_path = Path(frame["file_path"])
            image_path = dataset_dir / (
                frame_path if frame_path.suffix else frame_path.with_suffix(".png")
            )
            number = frame_number(frame_path)
            if dataset_index == base_dataset_index and number is not None:
                assigned_number = number
            else:
                assigned_number = next_frame_number
                next_frame_number += 1
            padding = max(
                padding,
                len(frame_path.stem) - len(frame_path.stem.rstrip("0123456789")),
                len(str(assigned_number)),
            )
            entries.append(
                (dataset_index, frame_path, image_path, frame, assigned_number)
            )

    staged_images = {}
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        for entry_index, (dataset_index, _, image_path, _, _) in enumerate(entries):
            if dataset_index == base_dataset_index:
                staged_image = temp_dir / f"{entry_index}{image_path.suffix}"
                shutil.move(str(image_path), str(staged_image))
                staged_images[entry_index] = staged_image

        for entry_index, (
            dataset_index,
            frame_path,
            image_path,
            frame,
            assigned_number,
        ) in enumerate(entries):
            output_frame_path = renumber_frame_path(
                frame_path, assigned_number, padding
            )
            output_image_path = output_dir / (
                output_frame_path
                if output_frame_path.suffix
                else output_frame_path.with_suffix(".png")
            )
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            if dataset_index == base_dataset_index:
                shutil.move(str(staged_images[entry_index]), str(output_image_path))
            else:
                shutil.copy2(image_path, output_image_path)

            combined_frame = dict(frame)
            combined_frame["file_path"] = output_frame_path.as_posix()
            combined_frames.append(combined_frame)

        combined_metadata["frames"] = combined_frames
        metadata_path = output_dir / f"transforms_{split}.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(combined_metadata, metadata_file, indent=2)
            metadata_file.write("\n")


def main():
    args = parse_args()
    dataset_dir = [path.expanduser() for path in args.dataset_dir]
    output_dir = dataset_dir[0]
    for split in ("train", "validation", "test"):
        combine_split(dataset_dir, output_dir, split)


if __name__ == "__main__":
    main()
