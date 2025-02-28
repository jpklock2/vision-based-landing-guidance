# Keypoints Detection

The Keypoints module processes a specified directory of images by unzipping them, cropping around the runway corner keypoints based on the ground truth CSV, and creating new annotations in the YOLO format. The module then trains a model using YOLOv8n-pose, saves the weights, and returns the trained model.

## How to Run

To execute the code, navigate to the `keypoints` folder and run the following command:

```bash
python get_keypoints.py /path/to/zip_images_dir.zip /path/to/ground_truth.csv
```

### Input Requirements

1. **Ground Truth CSV**:
   - The CSV file must contain four corner coordinates labeled as `x_A`, `y_A`, `x_B`, `y_B`, `x_C`, `y_C`, and `x_D`, `y_D`. The order of these coordinates is not important.
   - The CSV must also include an `image` column with the filename of each image.
   - This format is based on the LARD Dataset ground truth structure.

2. **Zip Images Directory**:
   - This should be a zip file containing all the images required for training.

### Output Structure

After running the script, the following directory structure will be created:

```
keypoints
├── train
│   ├── crop_data
│   ├── images
│   └── labels
├── unzipped_images
├── val
│   ├── crop_data
│   ├── images
│   └── labels
├── train_yaml_path.yaml
└── results
```

### Description of Output Directories

- **`train` and `val` Folders**:
  - `images`: Contains cropped images centered around the ground truth keypoints.
  - `crop_data`: Contains text files with cropping coordinates in the format:
    ```
    image_path left_coordinate upper_coordinate crop_window_width crop_window_height
    ```
    This data can be used to map the new ground truth coordinates after resizing.
  - `labels`: Contains YOLO annotation files for each image in the format:
    ```
    0 x_center y_center box_width box_height x1 y1 x2 y2 x3 y3 x4 y4
    ```
    - `x1`, `y1`: Top-left corner after resizing
    - `x2`, `y2`: Top-right corner after resizing
    - `x3`, `y3`: Bottom-left corner after resizing
    - `x4`, `y4`: Bottom-right corner after resizing

- **`unzipped_images`**:
  - Contains the unzipped original images.

- **`train_yaml_path.yaml`**:
  - YOLO configuration file used for training.

- **`results`**:
  - Stores the trained model weights and related outputs.
