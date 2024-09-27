import os
import csv
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import shutil

def create_yolo_annotations(df, image_dir, label_dir, subset_name):
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

    return x_center, y_center, box_width, box_height, [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]

  for i in df.index:
    filename = os.path.basename(df.loc[i, 'image'])
    src_image_path = os.path.join(image_dir, filename)

    if subset_name == 'train':
      dest_image_path = os.path.join(train_img_dir, filename)
      label_path = os.path.join(train_label_dir, os.path.splitext(filename)[0] + '.txt')
    else:
      dest_image_path = os.path.join(val_img_dir, filename)
      label_path = os.path.join(val_label_dir, os.path.splitext(filename)[0] + '.txt')

    xs = [df.loc[i, 'x_A'], df.loc[i, 'x_B'], df.loc[i, 'x_C'], df.loc[i, 'x_D']]
    ys = [df.loc[i, 'y_A'], df.loc[i, 'y_B'], df.loc[i, 'y_C'], df.loc[i, 'y_D']]

    x_center, y_center, width, height, normalized_keypoints = crop_image(src_image_path, dest_image_path, xs, ys)

    with open(label_path, 'a') as f:
      f.write(f"0 {x_center} {y_center} {width} {height} " + " ".join(map(str, normalized_keypoints)) + "\n")


def prep_data_for_keypoint_training(base_dir, ground_truth_df_dir, images_dir, destination_dir):

    train_img_dir = os.path.join(base_dir, 'train', 'images')
    train_label_dir = os.path.join(base_dir, 'train', 'labels')
    val_img_dir = os.path.join(base_dir, 'val', 'images')
    val_label_dir = os.path.join(base_dir, 'val', 'labels')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    lard_df = pd.read_csv(ground_truth_df_dir, delimiter=';')

    !unzip f"{images_dir}" -d f"{destination_dir}"

    train, test = train_test_split(lard_df, test_size=0.2, shuffle=True)
    create_yolo_annotations(train, f"{destination_dir}/images", train_label_dir, 'train')
    create_yolo_annotations(test, f"{destination_dir}/images", val_label_dir, 'val')


#prep_data_for_keypoint_training(
#    base_dir='/content/dataset', 
#    ground_truth_df_dir="/content/drive/MyDrive/Colab Notebooks/IC/LARD_train_DAAG_DIAP.csv",
#    images_dir="/content/drive/MyDrive/Colab Notebooks/IC/images.zip",
#    destination_dir="/content/lard_images/"
#    )
