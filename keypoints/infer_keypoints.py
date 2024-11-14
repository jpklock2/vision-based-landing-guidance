#!pip install ultralytics
import os
from PIL import Image
from ultralytics import YOLO

def main(pred_x, pred_y, pred_w, pred_h, image_path):

  def create_cropped_image_path(image_path):
    dir_name, file_name = os.path.split(image_path)
    cropped_dir = os.path.join(dir_name, 'cropped')
    os.makedirs(cropped_dir, exist_ok=True)
    return os.path.join(cropped_dir, file_name)

  def crop_image(percentage = 0.2):
    def calculate_coordinates(min_percentage):
      left = int (pred_x - min_percentage*pred_w)
      left = left if left > 0 else 0

      right = int (pred_x + pred_w + min_percentage*pred_w)
      right = right if right < width else width

      upper = int (pred_y - min_percentage*pred_h)
      upper = upper if upper > 0 else 0

      lower = int (pred_y+ pred_h + min_percentage*pred_h)
      lower = lower if lower < height else height
      return left, upper, right, lower

    im = Image.open(image_path)
    width, height = im.size
    left, upper, right, lower = calculate_coordinates(percentage/2)
    im = im.crop((left, upper, right, lower))
    im = im.resize((width, height))
    cropped_im_path = create_cropped_image_path(image_path)
    im.save(cropped_im_path)
    return cropped_im_path, [image_path, left, upper, abs(right-left), abs(upper-lower)]

  cropped_im_path, crop_data = crop_image()
  model = YOLO('/content/runs/train/exp/weights/best.pt')
  results = model.predict(source=cropped_im_path)
  keypoints = results[0].keypoints.xy

  return keypoints, crop_data