import argparse
from pathlib import Path
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
import random

def make_transform(seed=None):
    # Compose a set of mild/realistic augmentations for face images
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Blur(blur_limit=3, p=0.15),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.ToGray(p=0.05),
    ], p=1.0)

def augment_folder(input_dir: Path, output_dir: Path, per_image: int = 5, seed: int = None, skip_existing: bool = True):
    transform = make_transform(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    persons = [p for p in input_dir.iterdir() if p.is_dir()]
    for person in persons:
        out_person = output_dir / person.name
        out_person.mkdir(parents=True, exist_ok=True)
        imgs = [p for p in person.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for img_path in tqdm(imgs, desc=f"Aug {person.name}", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            base_name = img_path.stem
            # copy original (optionally)
            orig_out = out_person / f"{base_name}.jpg"
            if not (orig_out.exists() and skip_existing):
                cv2.imwrite(str(orig_out), img)

            for i in range(per_image):
                aug = transform(image=img)["image"]
                out_name = out_person / f"{base_name}_aug_{i+1}.jpg"
                # avoid overwriting existing augmented files if skip_existing
                if out_name.exists() and skip_existing:
                    continue
                cv2.imwrite(str(out_name), aug)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Augment face folders: faces/<person>/*.jpg")
    parser.add_argument("--input_dir", default=str(BASE_DIR / "faces"), help="Input faces folder (default: repository faces/)")
    parser.add_argument("--output_dir", default=str(BASE_DIR / "faces_aug"), help="Output augmented folder (default: repository faces_aug/)")
    parser.add_argument("--per_image", type=int, default=6, help="Augmentations per original image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_existing", action="store_true", help="Skip writing files that already exist")
    args = parser.parse_args()

    augment_folder(Path(args.input_dir), Path(args.output_dir), per_image=args.per_image, seed=args.seed, skip_existing=args.skip_existing)