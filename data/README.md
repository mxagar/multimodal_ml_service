# Datasets

This folder should contain the datasets locally, or links/pointers to them.

Currently, although not committed, the folder contains the images from the [Blur dataset @ Kaggle](https://www.kaggle.com/datasets/kwentar/blur-dataset), structured as follows:

```
kaggle_kwentar_blur_dataset/
    blur_dataset_resized_300/
        defocused_blurred/
        motion_blurred/
        sharp/
    blur_dataset_scaled/
    defocused_blurred/
    motion_blurred/
    sharp/
```

The resizing was carried out using the function `resize_images_in_folder()` from `src/domain/shared/image_utils.py`.
