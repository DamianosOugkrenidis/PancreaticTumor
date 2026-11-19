import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

CASE_SUFFIX = re.compile(r"[-_ ]?(pancreas|pancreas_seg|tumou?r|pda|lesion|segmentations?|seg|"
                         r"mask|label|volume|volumes|ct|img|image)$", re.IGNORECASE)
TUMOR_TOKENS = ("tumor", "tumour", "pda", "lesion")
PANCREAS_TOKENS = ("pancreas", "panc")
VOLUME_TOKENS = ("volume",)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image/label pairs from the raw dataset.")
    parser.add_argument("--raw", required=True, help="Folder with the extracted dataset.")
    parser.add_argument("--out", default="datasets/prepared", help="Output folder.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate the output folder.")
    return parser.parse_args()

def classify(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    folders = " ".join(part.lower() for part in relative.parts[:-1])
    name = relative.name.lower()

    if any(token in folders for token in TUMOR_TOKENS):
        return "tumor"
    if any(token in folders for token in PANCREAS_TOKENS):
        return "pancreas"
    if any(token in folders for token in VOLUME_TOKENS):
        return "volume"

    stem = re.sub(r"\.nii(\.gz)?$", "", name)
    tail = re.split(r"[-_ ]", stem)[-1]
    if tail in TUMOR_TOKENS:
        return "tumor"
    if tail in PANCREAS_TOKENS:
        return "pancreas"
    return None

def case_key(path: Path) -> str:
    stem = re.sub(r"\.nii(\.gz)?$", "", path.name)
    stem = re.sub(r"[-_]0{4}$", "", stem)
    for _ in range(4):
        stripped = CASE_SUFFIX.sub("", stem)
        if stripped == stem:
            break
        stem = stripped
    return re.sub(r"[^a-z0-9]", "", stem.lower())

def index_dataset(root: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {"volume": {}, "pancreas": {}, "tumor": {}}
    for path in sorted(root.rglob("*.nii*")):
        kind = classify(path, root)
        if kind is None:
            continue
        key = case_key(path)
        if not key:
            continue
        if key in index[kind]:
            print(f"  duplicate {kind} file for case {key}: {path.name} "
                  f"(keeping {index[kind][key].name})")
            continue
        index[kind][key] = path
    return index

def load_array(path: Path) -> tuple[np.ndarray, object]:
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D volume, got shape {data.shape}.")
    return data, image

def align_to_reference(source: np.ndarray, source_image, reference_image) -> np.ndarray:
    import nibabel as nib

    if source.shape == reference_image.shape and np.allclose(source_image.affine,
                                                             reference_image.affine, atol=1e-3):
        return source > 0
    resampled = nib.processing.resample_from_to(
        nib.Nifti1Image(source.astype(np.float32), source_image.affine), reference_image, order=0)
    return np.asanyarray(resampled.dataobj) > 0.5

def main() -> int:
    arguments = parse_arguments()
    raw = Path(arguments.raw)
    out = Path(arguments.out)
    if not raw.is_dir():
        raise SystemExit(f"Raw directory not found: {raw}")

    print(f"Scanning {raw.resolve()}")
    index = index_dataset(raw)
    volumes, pancreas, tumor = index["volume"], index["pancreas"], index["tumor"]
    print(f"  volumes: {len(volumes)} | pancreas masks: {len(pancreas)} | tumor masks: {len(tumor)}")

    if not volumes:
        print("\nNo CT volumes found. The 'Volume files.zip' archive (4.0 GiB) is required:")
        print('  python -m scripts.download_data --files "Volume files.zip"')
        return 1

    common = sorted(set(volumes) & set(pancreas) & set(tumor))
    report = {"prepared": [], "skipped": [], "overlap_voxels": {}, "empty_tumor": [],
              "missing_pancreas": sorted(set(volumes) - set(pancreas)),
              "missing_tumor": sorted(set(volumes) - set(tumor))}

    if not common:
        print("\nNo case has all three files. First keys of each category (for diagnosis):")
        print("  volumes :", list(volumes)[:5])
        print("  pancreas:", list(pancreas)[:5])
        print("  tumor   :", list(tumor)[:5])
        return 1

    image_dir = out / "imagesTr"
    label_dir = out / "labelsTr"
    if arguments.overwrite and out.exists():
        shutil.rmtree(out)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    import nibabel as nib

    entries = []
    for key in common:
        try:
            volume, volume_image = load_array(volumes[key])
            pancreas_mask, pancreas_image = load_array(pancreas[key])
            tumor_mask, tumor_image = load_array(tumor[key])

            pancreas_mask = align_to_reference(pancreas_mask, pancreas_image, volume_image)
            tumor_mask = align_to_reference(tumor_mask, tumor_image, volume_image)

            overlap = int(np.count_nonzero(pancreas_mask & tumor_mask))
            label = np.zeros(volume.shape, dtype=np.uint8)
            label[pancreas_mask] = 1
            label[tumor_mask] = 2

            if overlap:
                report["overlap_voxels"][key] = overlap
            if not np.any(label == 2):
                report["empty_tumor"].append(key)

            name = f"{key}_0000.nii.gz"
            image_out = nib.Nifti1Image(volume.astype(np.float32), volume_image.affine)
            image_out.header.set_data_dtype(np.float32)
            nib.save(image_out, str(image_dir / name))

            label_out = nib.Nifti1Image(label.astype(np.uint8), volume_image.affine)
            label_out.header.set_data_dtype(np.uint8)
            nib.save(label_out, str(label_dir / f"{key}.nii.gz"))
            entries.append({
                "image": f"imagesTr/{name}",
                "label": f"labelsTr/{key}.nii.gz",
                "shape": [int(v) for v in volume.shape],
                "pancreas_voxels": int(np.count_nonzero(label == 1)),
                "tumor_voxels": int(np.count_nonzero(label == 2)),
            })
            report["prepared"].append(key)
            print(f"  [{len(entries)}] {key}: shape {tuple(volume.shape)} | "
                  f"pancreas {int(np.count_nonzero(label == 1)):,} | "
                  f"tumor {int(np.count_nonzero(label == 2)):,} voxels")
        except Exception as error:
            report["skipped"].append({"case": key, "error": f"{type(error).__name__}: {error}"})
            print(f"  skipped {key}: {type(error).__name__}: {error}")

    dataset_json = {
        "name": "pancreatic_tumor_z5510650",
        "description": "Pancreas cancer segmentation on portal-venous phase CTs (Zenodo 5510650)",
        "labels": {"background": 0, "pancreas": 1, "tumor": 2},
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "numTraining": len(entries),
        "numTest": 0,
        "training": [{"image": entry["image"], "label": entry["label"]} for entry in entries],
        "test": [],
    }
    with (out / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
    with (out / "prepare_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    xlsx = next(raw.rglob("Metadata.xlsx"), None)
    if xlsx is not None:
        shutil.copy2(xlsx, out / "metadata.xlsx")

    print(f"\nPrepared {len(entries)} cases in {out.resolve()}")
    print(f"  skipped: {len(report['skipped'])} | cases without tumor: "
          f"{len(report['empty_tumor'])} | cases with pancreas/tumor overlap: "
          f"{len(report['overlap_voxels'])}")
    if report["skipped"]:
        print("  see prepare_report.json for details")
    print("\nTraining example:")
    print("  python -m scripts.train --config configs/unetr.yml")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
