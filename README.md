# MS COCO 2014

Microsoft Common Objects in Context (MS COCO) dataset for image captioning.

**Official site**: https://cocodataset.org/

## Download & Extract

```bash
# Images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Karpathy splits
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

# Extract
unzip caption_datasets.zip dataset_coco.json -d karpathy-split
unzip '*2014.zip'

# Cleaning
rm *.zip
```

## Directory Structure

```
COCO-2014/
├── train2014/          # 82,783 training images
├── val2014/            # 40,504 validation images
├── test2014/           # 40,775 test images (no annotations)
├── annotations/        # Official COCO annotations
└── karpathy-splits/    # Karpathy split for image captioning
    ├── dataset_coco.json
    ├── coco_karpathy_dataset.py
    └── README.md
```

## Karpathy Split

For image captioning, use the Karpathy split (see `karpathy-splits/README.md`):

| Split           | Images  |
| --------------- | ------- |
| train + restval | 113,287 |
| val             | 5,000   |
| test            | 5,000   |

## Reference

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={ECCV},
  year={2014}
}
```
