# MS COCO Karpathy Splits

Karpathy split for image captioning (Karpathy & Li, 2015).

**Source**: [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/deepimagesent/)

## Splits

| Split     | Images      | Source    |
| --------- | ----------- | --------- |
| train     | 82,783      | train2014 |
| restval   | 30,504      | val2014   |
| val       | 5,000       | val2014   |
| test      | 5,000       | val2014   |
| **Total** | **123,287** |           |

For training, use `train + restval` (113,287 images).

## Usage

```python
from coco_karpathy_dataset import COCOKarpathyDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

root_dir = "/path/to/COCO-2014"
json_path = "/path/to/dataset_coco.json"

# Full training set (113,287 images)
train_set = COCOKarpathyDataset(root_dir, json_path, split=["train", "restval"], transform=transform)

# Validation set (5,000 images)
val_set = COCOKarpathyDataset(root_dir, json_path, split="val", transform=transform)

# Test set (5,000 images)
test_set = COCOKarpathyDataset(root_dir, json_path, split="test", transform=transform)
```

## Run Test

```bash
uv run coco_karpathy_dataset.py
```

## References

```bibtex
@inproceedings{karpathy2015deep,
  title={Deep Visual-Semantic Alignments for Generating Image Descriptions},
  author={Karpathy, Andrej and Li, Fei-Fei},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and others},
  booktitle={ECCV},
  year={2014}
}
```
