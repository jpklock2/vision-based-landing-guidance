#!pip install ultralytics
import os
import csv
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import shutil
import zipfile
import argparse
from ultralytics import YOLO

file_path  = os.path.dirname(os.path.abspath(__file__))

def create_yolo_annotations(df, data_dir, dest_dir, crop_dir):
  image_dir, label_dir = data_dir
  dest_image_dir, dest_label_dir = dest_dir

  def crop_image(image_path, dest_image_path, xs, ys, percentage = 0.2):
    im = Image.open(image_path)
    width, height = im.size
    min_percentage = percentage/2

    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

    original_bounding_box_width = abs(x_min - x_max)
    original_bounding_box_height = abs(y_min - y_max)

    left = int (x_min - min_percentage*original_bounding_box_width)
    left = left if left > 0 else 0

    right = int (x_max + min_percentage*original_bounding_box_width)
    right = right if right < width else width

    upper = int (y_min - min_percentage*original_bounding_box_height)
    upper = upper if upper > 0 else 0

    lower = int (y_max + min_percentage*original_bounding_box_height)
    lower = lower if lower < height else height

    crop_data = [image_path, left, upper, abs(right-left), abs(upper-lower)]
    
    im = im.crop((left, upper, right, lower))
    cropped_im_width, cropped_img_height = im.size

    box_width = original_bounding_box_width * (1/cropped_im_width)
    box_height = original_bounding_box_height * (1/cropped_img_height)

    ### For keypoints
    sorted_xs = list(xs)
    x_a, x_b, x_c, x_d = sorted_xs[0], sorted_xs[1], sorted_xs[2], sorted_xs[3]

    sorted_ys = list(ys)
    y_a, y_b, y_c, y_d = sorted_ys[0], sorted_ys[1], sorted_ys[2], sorted_ys[3]

    runnaway_initial_top_width = x_b - x_a
    runnaway_initial_end_hight = y_b - y_d
    runnaway_initial_bottom_width = x_d - x_c
    runnaway_initial_start_hight = y_a - y_c

    x_1 = abs(x_a - left) * (1/cropped_im_width)
    x_3 = abs(x_c - left) * (1/cropped_im_width)
    x_2 = x_1 + (runnaway_initial_top_width * (1/cropped_im_width))
    x_4 = x_3 + (runnaway_initial_bottom_width * (1/cropped_im_width))

    y_1 = abs(y_a - upper) * (1/cropped_img_height)
    y_2 = abs(y_b - upper) * (1/cropped_img_height)
    y_3 = y_1 - (runnaway_initial_start_hight * (1/cropped_img_height))
    y_4 = y_2 - (runnaway_initial_end_hight * (1/cropped_img_height))
    im = im.resize((width, height))
    im.save(dest_image_path)

    new_xs = [x_1, x_2, x_3, x_4]
    new_ys = [y_1, y_2, y_3, y_4]

    x_center = min(new_xs) + (box_width /2)
    y_center = min(new_ys) + (box_height/2)

    return x_center, y_center, box_width, box_height, [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4], crop_data

  all_crop_data = []
  for i in df.index:
    filename = os.path.basename(df.loc[i, 'image'])
    src_image_path = os.path.join(image_dir, filename)

    dest_image_path = os.path.join(dest_image_dir, filename)
    label_path = os.path.join(dest_label_dir, os.path.splitext(filename)[0] + '.txt')

    crop_path = os.path.join(crop_dir, os.path.splitext(filename)[0] + '.txt')

    xs = [df.loc[i, 'x_A'], df.loc[i, 'x_B'], df.loc[i, 'x_C'], df.loc[i, 'x_D']]
    ys = [df.loc[i, 'y_A'], df.loc[i, 'y_B'], df.loc[i, 'y_C'], df.loc[i, 'y_D']]

    x_center, y_center, width, height, normalized_keypoints, crop_data = crop_image(src_image_path, dest_image_path, xs, ys)
    all_crop_data.append(crop_data)

    with open(label_path, 'a') as f:
      f.write(f"0 {x_center} {y_center} {width} {height} " + " ".join(map(str, normalized_keypoints)) + "\n")

  output_file = os.path.join(crop_dir, os.path.basename("crop_data"))
  with open(output_file, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(all_crop_data)

def prep_data_for_keypoint_training(ground_truth_df_dir, images_dir, train_img_dir, val_img_dir):
    train_label_dir = os.path.join(file_path, 'train', 'labels')
    train_crop_dir = os.path.join(file_path, 'train', 'crop_data')
    val_label_dir = os.path.join(file_path, 'val', 'labels')
    val_crop_dir = os.path.join(file_path, 'val', 'crop_data')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(train_crop_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(val_crop_dir, exist_ok=True)

    lard_df = pd.read_csv(ground_truth_df_dir, delimiter=';')

    destination_dir = os.path.join(file_path, 'unzipped_images')
    os.makedirs(destination_dir, exist_ok=True)

    with zipfile.ZipFile(images_dir, 'r') as zip_ref:
      zip_ref.extractall(destination_dir)

    train, test = train_test_split(lard_df, test_size=0.2, shuffle=True)
    create_yolo_annotations(train, [f"{destination_dir}/images", train_label_dir], [train_img_dir, train_label_dir], train_crop_dir)
    create_yolo_annotations(test, [f"{destination_dir}/images", val_label_dir], [val_img_dir, val_label_dir], val_crop_dir)

def train_keypoints(train_img_dir, val_img_dir):
    data_yaml_content = f"""
    train: {train_img_dir}
    val: {val_img_dir}

    nc: 1
    names: ['runnaway']

    kpt_shape: [4, 2]
    """

    train_yaml_path = os.path.join(file_path, 'train_yaml_path.yaml')
    with open(train_yaml_path, 'w') as file:
        file.write(data_yaml_content)
    
    
    train_weights_dir = os.path.join(file_path, 'results')
    os.makedirs(train_weights_dir, exist_ok=True)

    model = YOLO('yolov8n-pose.yaml')
    model.train(data=train_yaml_path, epochs=1, imgsz=640, batch=16, workers=2, project=train_weights_dir, name='exp', exist_ok=True)
    return model


def get_keypoints(ground_truth_df_dir, images_dir):
    train_img_dir = os.path.join(file_path, 'train', 'images')
    val_img_dir = os.path.join(file_path, 'val', 'images')

    prep_data_for_keypoint_training(
        ground_truth_df_dir= ground_truth_df_dir,
        images_dir= images_dir,
        train_img_dir = train_img_dir,
        val_img_dir = val_img_dir
    )
    model = train_keypoints(
        train_img_dir = train_img_dir,
        val_img_dir = val_img_dir
    )
    return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Prep data for yolo annotation, crops it around the ground truth and train it")
  parser.add_argument("zip_images_dir", type=str, help="Path to the zip folder that contains the images")
  parser.add_argument("ground_truth_dir", type=str, help="Path to the csv file that contains the images annotations")

  args = parser.parse_args()
  model = get_keypoints(args.ground_truth_dir, args.zip_images_dir)