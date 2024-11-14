#!pip install ultralytics
import os
import argparse
from PIL import Image
from ultralytics import YOLO


def create_cropped_image_path(image_path):
  dir_name, file_name = os.path.split(image_path)
  cropped_dir = os.path.join(dir_name, 'cropped')
  os.makedirs(cropped_dir, exist_ok=True)
  return os.path.join(cropped_dir, file_name)


def calculate_coordinates(pred_x, pred_y, pred_w, pred_h, width, height, min_percentage):
  left = int (pred_x - min_percentage*pred_w)
  left = left if left > 0 else 0

  right = int (pred_x + pred_w + min_percentage*pred_w)
  right = right if right < width else width

  upper = int (pred_y - min_percentage*pred_h)
  upper = upper if upper > 0 else 0

  lower = int (pred_y + pred_h + min_percentage*pred_h)
  lower = lower if lower < height else height
  return left, upper, right, lower


def crop_image(pred_x, pred_y, pred_w, pred_h, image_path, percentage = 0.2):
  im = Image.open(image_path)
  width, height = im.size
  left, upper, right, lower = calculate_coordinates(pred_x, pred_y, pred_w, pred_h, width, height, percentage/2)
  cropped_im_path = create_cropped_image_path(image_path)
  im = im.crop((left, upper, right, lower)).resize((width, height)).save(cropped_im_path)
  im = im.resize((width, height))
  im.save(cropped_im_path)
  return cropped_im_path, [image_path, left, upper, abs(right-left), abs(upper-lower)]

def main_keypoints(pred_x, pred_y, pred_w, pred_h, image_path):
  cropped_im_path, crop_data = crop_image(pred_x, pred_y, pred_w, pred_h, image_path)
  model_path = os.path.join("/mnt/d/Master/Airport_Runway_Detection/vision-based-landing-guidance", "models", "keypoints", "model.pt")
  model = YOLO(model_path)
  results = model.predict(source=cropped_im_path)
  keypoints = results[0].keypoints.xy

  return keypoints, crop_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the image around the tracking detection and predict keypoints")
    parser.add_argument("pred_x", type=int, help="Predicted x coordinate")
    parser.add_argument("pred_y", type=int, help="Predicted y coordinate")
    parser.add_argument("pred_w", type=int, help="Predicted width")
    parser.add_argument("pred_h", type=int, help="Predicted height")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    keypoints, crop_data = main_keypoints(args.pred_x, args.pred_y, args.pred_w, args.pred_h, args.image_path)
