# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow",
# ]
# ///

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCOKarpathyDataset(Dataset):
    """MS COCO Captions dataset with Karpathy split (Karpathy & Li, 2015).

    Loads images and captions from dataset_coco.json.

    Args:
        root_dir: MS COCO root directory (contains train2014/, val2014/)
        json_path: Path to dataset_coco.json
        split: 'train', 'val', 'test', 'restval', or list like ['train', 'restval']
        transform: Image transforms

    Splits:
        train: 82,783 images (from train2014)
        restval: 30,504 images (from val2014, merged into train for full 113,287)
        val: 5,000 images (from val2014)
        test: 5,000 images (from val2014)

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize(256),
        ...     transforms.CenterCrop(256),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ... ])
        >>> # Full training set (113,287 images)
        >>> train_set = COCOKarpathyDataset(root, json_path, split=['train', 'restval'], transform=transform)
        >>> # Test set (5,000 images)
        >>> test_set = COCOKarpathyDataset(root, json_path, split='test', transform=transform)
    """

    def __init__(self, root_dir, json_path, split="test", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(json_path) as f:
            data = json.load(f)

        if isinstance(split, str):
            split = [split]

        self.images = [img for img in data["images"] if img["split"] in split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]

        # filepath is 'train2014' or 'val2014'
        img_path = Path(self.root_dir) / img_info["filepath"] / img_info["filename"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get all captions and randomly select one
        captions = [s["raw"] for s in img_info["sentences"]]
        caption = captions[torch.randint(0, len(captions), (1,)).item()]

        return image, caption


if __name__ == "__main__":
    from torchvision import transforms

    # Relative to this file's location
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent  # COCO-2014/
    json_path = script_dir / "dataset_coco.json"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Test all splits
    for split in ["train", "restval", "val", "test"]:
        ds = COCOKarpathyDataset(root_dir, json_path, split=split, transform=transform)
        print(f"{split}: {len(ds):,}")

    # Full training set
    train_full = COCOKarpathyDataset(
        root_dir, json_path, split=["train", "restval"], transform=transform
    )
    print(f"train + restval: {len(train_full):,}")
