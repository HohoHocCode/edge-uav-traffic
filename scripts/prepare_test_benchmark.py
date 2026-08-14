#!/usr/bin/env python3
"""Prepare the four VisDrone test tasks used by this repository.

The input shipped in ``test/`` is sequence based.  Task 1 needs an image
split, while task 5 needs frame-level counts.  This script derives both from
the VID data and deliberately leaves the original VID and MOT directories
untouched:

* Task 1 / DET: take one frame every ``--sample-step`` frames in each sequence
  and convert the 10-column sequence annotation to the 8-column DET format.
* Task 2 / VID: keep every source frame and annotation as-is.
* Task 4 / MOT: keep every source frame, track id and annotation as-is.
* Task 5 / Crowd Counting: write ground-truth counts for every class/frame.

Images in the derived DET split are hard-linked when possible, so preparing a
benchmark does not duplicate the JPEG payload.  ``--link-mode copy`` is
available for filesystems that do not support hard links.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Annotation:
    frame: int
    target_id: int
    x: float
    y: float
    width: float
    height: float
    score: float
    category: int
    truncation: str
    occlusion: str
    det_fields: tuple[str, ...]

    @property
    def evaluated_class(self) -> int | None:
        if (
            self.width <= 0
            or self.height <= 0
            or self.score == 0
            or not 1 <= self.category <= 10
        ):
            return None
        return self.category - 1


def parse_sequence_annotation(path: Path) -> dict[int, list[Annotation]]:
    """Read VisDrone VID/MOT rows grouped by 1-based frame index."""
    per_frame: dict[int, list[Annotation]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().rstrip(",").split(",")
            if fields == [""]:
                continue
            if len(fields) < 10:
                raise ValueError(
                    f"{path}:{line_number}: expected 10 columns, got {len(fields)}"
                )
            try:
                item = Annotation(
                    frame=int(float(fields[0])),
                    target_id=int(float(fields[1])),
                    x=float(fields[2]),
                    y=float(fields[3]),
                    width=float(fields[4]),
                    height=float(fields[5]),
                    score=float(fields[6]),
                    category=int(float(fields[7])),
                    truncation=fields[8],
                    occlusion=fields[9],
                    det_fields=tuple(fields[2:10]),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
            if item.frame < 1:
                raise ValueError(f"{path}:{line_number}: frame index must be >= 1")
            per_frame[item.frame].append(item)
    return dict(per_frame)


def frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"frame filename must be numeric: {path}") from exc


def sequence_inventory(root: Path) -> tuple[list[tuple[str, list[Path], Path]], list[str]]:
    sequence_root = root / "sequences"
    annotation_root = root / "annotations"
    if not sequence_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            f"expected sequences/ and annotations/ under {root}"
        )

    sequences: list[tuple[str, list[Path], Path]] = []
    for sequence_dir in sorted(p for p in sequence_root.iterdir() if p.is_dir()):
        images = sorted(
            (p for p in sequence_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            key=frame_index,
        )
        annotation = annotation_root / f"{sequence_dir.name}.txt"
        if not images:
            continue
        if not annotation.is_file():
            raise FileNotFoundError(
                f"sequence {sequence_dir.name} has frames but no annotation: {annotation}"
            )
        indices = [frame_index(p) for p in images]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate frame index in {sequence_dir}")
        sequences.append((sequence_dir.name, images, annotation))

    frame_names = {name for name, _, _ in sequences}
    annotation_only = sorted(
        p.stem for p in annotation_root.glob("*.txt") if p.stem not in frame_names
    )
    if not sequences:
        raise ValueError(f"no usable sequences found under {root}")
    return sequences, annotation_only


def tree_fingerprint(root: Path) -> str:
    """Cheap inventory fingerprint used to prove source trees were not changed."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        record = f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise FileExistsError(
                f"refusing to replace a different existing file: {destination}"
            )
        try:
            return "hardlink" if os.path.samefile(source, destination) else "copy"
        except OSError:
            return "existing"

    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def prepare(args: argparse.Namespace) -> dict:
    test_root = args.test_root.resolve()
    vid_root = (args.vid_root or test_root / "VisDrone2019-VID-test-dev").resolve()
    mot_root = (args.mot_root or test_root / "VisDrone2019-MOT-test-dev").resolve()
    det_root = (args.det_root or test_root / "VisDrone2019-DET-test-dev").resolve()
    count_root = (
        args.count_root or test_root / "VisDrone2019-CC-test-dev"
    ).resolve()

    if args.sample_step < 1:
        raise ValueError("--sample-step must be >= 1")
    if not 1 <= args.sample_offset <= args.sample_step:
        raise ValueError("--sample-offset must be in [1, sample-step]")
    if not mot_root.is_dir():
        raise FileNotFoundError(f"MOT root does not exist: {mot_root}")
    for output in (det_root, count_root):
        if output in {vid_root, mot_root}:
            raise ValueError("derived output must not overwrite VID or MOT source data")

    source_before = {
        "vid": tree_fingerprint(vid_root),
        "mot": tree_fingerprint(mot_root),
    }
    sequences, annotation_only = sequence_inventory(vid_root)
    mot_sequences, mot_annotation_only = sequence_inventory(mot_root)

    det_images = det_root / "images"
    det_annotations = det_root / "annotations"
    det_images.mkdir(parents=True, exist_ok=True)
    det_annotations.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict] = []
    count_rows: list[dict] = []
    sequence_rows: list[dict] = []
    expected_det_names: set[str] = set()
    class_totals = [0] * len(CLASS_NAMES)

    for sequence, images, annotation_path in sequences:
        annotations = parse_sequence_annotation(annotation_path)
        image_indices = {frame_index(path) for path in images}
        unknown_frames = sorted(set(annotations) - image_indices)
        if unknown_frames:
            print(
                f"[warn] {sequence}: {len(unknown_frames)} annotated frame(s) "
                "have no image and are excluded",
                file=sys.stderr,
            )

        per_sequence_totals = [0] * len(CLASS_NAMES)
        selected = images[args.sample_offset - 1 :: args.sample_step]
        for image in selected:
            index = frame_index(image)
            items = annotations.get(index, [])
            destination_stem = f"{sequence}_{image.stem}"
            destination_image = det_images / f"{destination_stem}{image.suffix.lower()}"
            destination_annotation = det_annotations / f"{destination_stem}.txt"
            storage = link_or_copy(image, destination_image, args.link_mode)
            det_text = "".join(",".join(item.det_fields) + "\n" for item in items)
            write_text(destination_annotation, det_text)
            expected_det_names.add(destination_image.name)

            valid = sum(item.evaluated_class is not None for item in items)
            selection_rows.append(
                {
                    "image_id": destination_stem,
                    "sequence": sequence,
                    "frame_index": index,
                    "source_image": relative(image, test_root),
                    "image": relative(destination_image, test_root),
                    "annotation": relative(destination_annotation, test_root),
                    "n_rows": len(items),
                    "n_objects": valid,
                    "n_excluded": len(items) - valid,
                    "storage": storage,
                }
            )

        for image in images:
            index = frame_index(image)
            counts = [0] * len(CLASS_NAMES)
            items = annotations.get(index, [])
            for item in items:
                class_index = item.evaluated_class
                if class_index is not None:
                    counts[class_index] += 1
                    per_sequence_totals[class_index] += 1
                    class_totals[class_index] += 1

            row = {
                "sequence": sequence,
                "frame_index": index,
                "image": relative(image, test_root),
            }
            row.update(dict(zip(CLASS_NAMES, counts)))
            row["person_total"] = counts[0] + counts[1]
            row["vehicle_total"] = sum(counts[2:])
            row["total"] = sum(counts)
            row["excluded_regions"] = len(items) - sum(counts)
            count_rows.append(row)

        sequence_row: dict[str, object] = {
            "sequence": sequence,
            "n_frames": len(images),
        }
        for class_name, total in zip(CLASS_NAMES, per_sequence_totals):
            sequence_row[f"{class_name}_sum"] = total
            sequence_row[f"{class_name}_mean"] = f"{total / len(images):.6f}"
        sequence_row["total_sum"] = sum(per_sequence_totals)
        sequence_row["total_mean"] = f"{sum(per_sequence_totals) / len(images):.6f}"
        sequence_rows.append(sequence_row)

    actual_det_names = {
        p.name for p in det_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual_det_names != expected_det_names:
        stale = sorted(actual_det_names - expected_det_names)
        missing = sorted(expected_det_names - actual_det_names)
        raise RuntimeError(
            "DET output does not match this sampling configuration; "
            f"stale={stale[:5]} missing={missing[:5]}. "
            "Move the old generated DET directory away and rerun."
        )
    actual_ann_names = {p.stem for p in det_annotations.glob("*.txt")}
    expected_ann_names = {Path(name).stem for name in expected_det_names}
    if actual_ann_names != expected_ann_names:
        raise RuntimeError("DET images and annotations do not have identical stems")

    selection_fields = [
        "image_id", "sequence", "frame_index", "source_image", "image",
        "annotation", "n_rows", "n_objects", "n_excluded", "storage",
    ]
    write_csv(det_root / "selection.csv", selection_fields, selection_rows)

    count_fields = [
        "sequence", "frame_index", "image", *CLASS_NAMES,
        "person_total", "vehicle_total", "total", "excluded_regions",
    ]
    write_csv(count_root / "counts_by_frame.csv", count_fields, count_rows)

    sequence_fields = ["sequence", "n_frames"]
    for class_name in CLASS_NAMES:
        sequence_fields.extend([f"{class_name}_sum", f"{class_name}_mean"])
    sequence_fields.extend(["total_sum", "total_mean"])
    write_csv(count_root / "counts_by_sequence.csv", sequence_fields, sequence_rows)

    class_map = {
        "visdrone_category_to_model_class": {
            str(index + 1): {"class_id": index, "name": name}
            for index, name in enumerate(CLASS_NAMES)
        },
        "excluded": {
            "category_0": "ignored region",
            "category_11": "others (outside the evaluated taxonomy)",
            "score_0": "excluded by the VisDrone protocol",
            "non_positive_box": "invalid width or height",
        },
    }
    write_text(
        count_root / "classes.json",
        json.dumps(class_map, indent=2, ensure_ascii=False) + "\n",
    )

    source_after = {
        "vid": tree_fingerprint(vid_root),
        "mot": tree_fingerprint(mot_root),
    }
    if source_before != source_after:
        raise RuntimeError("source VID/MOT data changed during preparation")

    total_frames = sum(len(images) for _, images, _ in sequences)
    manifest = {
        "schema_version": 1,
        "class_names": list(CLASS_NAMES),
        "source_trees_unchanged": True,
        "source_inventory_sha256": source_after,
        "tasks": {
            "task1_detection": {
                "root": relative(det_root, test_root),
                "images": relative(det_images, test_root),
                "annotations": relative(det_annotations, test_root),
                "selection": relative(det_root / "selection.csv", test_root),
                "sampling": {
                    "unit": "sequence",
                    "step": args.sample_step,
                    "offset_1_based": args.sample_offset,
                    "rule": (
                        "sort frames numerically within each sequence, then take "
                        f"frames[{args.sample_offset - 1}::{args.sample_step}]"
                    ),
                },
                "n_images": len(selection_rows),
            },
            "task2_vid": {
                "root": relative(vid_root, test_root),
                "policy": "all frames and original annotations, unchanged",
                "benchmark": (
                    "persistent tracking within each sequence, all 10 classes; "
                    "frame-level detection AP is diagnostic only"
                ),
                "n_sequences": len(sequences),
                "n_frames": total_frames,
            },
            "task4_mot": {
                "root": relative(mot_root, test_root),
                "policy": "all frames, track ids and original annotations, unchanged",
                "benchmark": "persistent tracking, official VisDrone MOT 5-class policy",
                "n_sequences": len(mot_sequences),
                "n_frames": sum(len(images) for _, images, _ in mot_sequences),
            },
            "task5_crowd_counting": {
                "root": relative(count_root, test_root),
                "ground_truth_by_frame": relative(
                    count_root / "counts_by_frame.csv", test_root
                ),
                "summary_by_sequence": relative(
                    count_root / "counts_by_sequence.csv", test_root
                ),
                "n_frames": len(count_rows),
                "class_instance_totals": dict(zip(CLASS_NAMES, class_totals)),
                "counting_rule": (
                    "count category 1..10 with score != 0 and positive box size; "
                    "category 0, category 11, score 0 and invalid boxes are excluded"
                ),
            },
        },
        "annotation_files_without_images": {
            "vid": annotation_only,
            "mot": mot_annotation_only,
        },
    }
    manifest_path = test_root / "benchmark_manifest.json"
    write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def verify(test_root: Path) -> dict:
    manifest_path = test_root / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]

    det = tasks["task1_detection"]
    images = list((test_root / det["images"]).glob("*"))
    annotations = list((test_root / det["annotations"]).glob("*.txt"))
    image_stems = {p.stem for p in images if p.suffix.lower() in IMAGE_SUFFIXES}
    annotation_stems = {p.stem for p in annotations}
    if image_stems != annotation_stems or len(image_stems) != det["n_images"]:
        raise RuntimeError("Task 1 verification failed: image/annotation mismatch")

    count = tasks["task5_crowd_counting"]
    count_path = test_root / count["ground_truth_by_frame"]
    with count_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != count["n_frames"]:
        raise RuntimeError("Task 5 verification failed: frame count mismatch")
    totals = {name: sum(int(row[name]) for row in rows) for name in CLASS_NAMES}
    if totals != count["class_instance_totals"]:
        raise RuntimeError("Task 5 verification failed: per-class totals mismatch")

    current_fingerprints = {
        "vid": tree_fingerprint(test_root / tasks["task2_vid"]["root"]),
        "mot": tree_fingerprint(test_root / tasks["task4_mot"]["root"]),
    }
    if current_fingerprints != manifest["source_inventory_sha256"]:
        raise RuntimeError("source VID/MOT data no longer matches the manifest")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=repository_root / "test")
    parser.add_argument("--vid-root", type=Path, default=None)
    parser.add_argument("--mot-root", type=Path, default=None)
    parser.add_argument("--det-root", type=Path, default=None)
    parser.add_argument("--count-root", type=Path, default=None)
    parser.add_argument("--sample-step", type=int, default=100)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=1,
        help="1-based position selected from each sample-step block (default: 1)",
    )
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="validate existing generated data"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        manifest = verify(args.test_root.resolve())
    else:
        prepare(args)
        manifest = verify(args.test_root.resolve())

    tasks = manifest["tasks"]
    print(
        "[ok] benchmark ready: "
        f"DET={tasks['task1_detection']['n_images']} sampled images, "
        f"VID={tasks['task2_vid']['n_frames']} full frames, "
        "MOT=unchanged, "
        f"CrowdCounting={tasks['task5_crowd_counting']['n_frames']} frames"
    )
    print(f"[ok] {args.test_root.resolve() / 'benchmark_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
