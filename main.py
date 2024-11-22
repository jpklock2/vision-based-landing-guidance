import os
import sys
import shutil
import argparse
import pandas as pd

sys.path.insert(0, "LoRAT")

from LoRAT.main_remodeled import main_tracking
from keypoints.infer_keypoints import main_keypoints
from pose_estimation.pose_estimation_utils import format_keypoints
from pose_estimation.pose_estimation import main_pose_estimation
from utils.validations import validate_sequence


def setup_arg_parser():
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('--input_dir', type=str, help='Directory with images')
    parser.add_argument('--output_dir', type=str, help='Directory to store results')
    return parser


if __name__ == '__main__':

    parser = setup_arg_parser()
    args = parser.parse_args()
    input_dir = args.input_dir

    airport_name = "CYUL"
    runways = ["06L"]

    validate_sequence(input_dir)

    shutil.rmtree(os.path.join("LoRAT", "trackit", "datasets", "cache"), ignore_errors=True)
    tracking_output_dir = main_tracking(parser)
    # tracking_output_dir = "/mnt/d/Master/Airport_Runway_Detection/vision-based-landing-guidance/outputs/2024.11.22-21.23.00-986630"

    sequence_name = os.path.basename(input_dir)
    csv_path = os.path.join(tracking_output_dir, "eval", "epoch_0", "results", "DINOv2-B-224", "MyDataset-test", sequence_name, "eval.csv")
    eval_df = pd.read_csv(csv_path, sep=",")

    estimated_poses_list = []
    for i in range(len(eval_df)):
        image_df = eval_df.iloc[i]
        image_name = str(int(image_df["# ind"]) + 1).zfill(4) + ".jpg"
        image_path = os.path.join(input_dir, "frames", image_name)
        pred_x = image_df.pred_x
        pred_y = image_df.pred_y
        pred_w = image_df.pred_w
        pred_h = image_df.pred_h

        # pred_x, pred_y, pred_w, pred_h, image_path
        keypoints, crop_data = main_keypoints(pred_x, pred_y, pred_w, pred_h, image_path)
        bbox_coord = format_keypoints(keypoints, crop_data)
        estimated_pose, estimated_distance, pose_errors, distance_errors = main_pose_estimation(airport_name, runways, bbox_coord, csv_file_path=None)
        estimated_poses_list.append(estimated_pose + estimated_distance)
        a = 1
