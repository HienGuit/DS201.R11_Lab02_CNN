from pathlib import Path
import hashlib

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_rgb(path):
    with Image.open(path) as image:
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        return image.convert("RGB")


def build_transforms(IMG_SIZE, augmentation="standard"):
    if augmentation == "standard":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif augmentation == "light":
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif augmentation == "none":
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        raise ValueError("augmentation phải là standard, light hoặc none")

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def make_loaders(root, cfg):
    DATA_ROOT = Path(root)
    IMG_SIZE = cfg["img_size"]
    BATCH = cfg["batch_size"]
    SEED = cfg["seed"]
    num_workers = cfg.get("num_workers", 2)

    if not (DATA_ROOT / "train").is_dir():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_ROOT / 'train'}")

    train_tf, eval_tf = build_transforms(
        IMG_SIZE, augmentation=cfg.get("augmentation", "standard")
    )

    full_dataset = datasets.ImageFolder(
        DATA_ROOT / "train", transform=train_tf, loader=load_rgb
    )
    CLASSES = full_dataset.classes
    if len(CLASSES) != cfg.get("num_classes", 21):
        raise ValueError(f"Số lớp không khớp: tìm thấy {len(CLASSES)} lớp")

    if (DATA_ROOT / "val").is_dir():
        train_ds = full_dataset
        val_ds = datasets.ImageFolder(
            DATA_ROOT / "val", transform=eval_tf, loader=load_rgb
        )
        if full_dataset.class_to_idx != val_ds.class_to_idx:
            raise ValueError("Train/validation class mapping differs")
        split_source = "existing_val"
    else:
        val_ratio = cfg.get("val_ratio", 0.2)
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio phải nằm giữa 0 và 1")
        train_size = int((1 - val_ratio) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_split = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED),
        )

        val_dataset = datasets.ImageFolder(
            DATA_ROOT / "train", transform=eval_tf, loader=load_rgb
        )
        if full_dataset.samples != val_dataset.samples:
            raise ValueError("Thứ tự ảnh của hai dataset không khớp")
        val_ds = Subset(val_dataset, val_split.indices)
        split_source = "split_from_train"

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Tập train và validation phải có ảnh")

    train_dl = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=num_workers,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=num_workers,
    )

    def image_list(dataset):
        if isinstance(dataset, Subset):
            indices, base = dataset.indices, dataset.dataset
        else:
            indices, base = range(len(dataset)), dataset
        return [
            {"path": str(Path(base.samples[i][0]).relative_to(DATA_ROOT)),
             "label": base.samples[i][1]}
            for i in indices
        ]

    manifest = {
        "classes": CLASSES, "seed": SEED,
        "val_ratio": cfg.get("val_ratio", 0.2), "source": split_source,
        "train": image_list(train_ds), "val": image_list(val_ds),
        "counts": {"train": len(train_ds), "val": len(val_ds)},
    }
    train_paths = {row["path"] for row in manifest["train"]}
    val_paths = {row["path"] for row in manifest["val"]}
    if train_paths & val_paths:
        raise ValueError("Ảnh train và validation bị trùng")
    encoded = "\n".join(
        f"{phase}:{row['path']}:{row['label']}"
        for phase in ("train", "val") for row in manifest[phase]
    )
    manifest["split_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return train_dl, val_dl, CLASSES, manifest, eval_tf


def test_loader(root, cfg, classes, eval_tf):
    test_ds = datasets.ImageFolder(
        Path(root) / "test", transform=eval_tf, loader=load_rgb
    )
    if test_ds.class_to_idx != {name: i for i, name in enumerate(classes)}:
        raise ValueError("Train/test class mapping differs")
    test_dl = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 2),
    )
    return test_dl, test_ds
