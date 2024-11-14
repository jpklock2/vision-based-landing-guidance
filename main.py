import os
import argparse
import pandas as pd

# from LoRAT.main_remodeled import main_tracking
from keypoints.infer_keypoints import main_keypoints


def setup_arg_parser():
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('--input_dir', type=str, help='Directory with images')
    parser.add_argument('--output_dir', type=str, help='Directory to store results')
    return parser


if __name__ == '__main__':

    parser = setup_arg_parser()
    args = parser.parse_args()
    input_dir = args.input_dir
    # "/mnt/d/Master/Airport_Runway_Detection/vision-based-landing-guidance/inputs/CYUL_06L_35_test/frames"

    # main_tracking()

    # Unzip dos results
    # Get eval.csv path

    csv_path = "/mnt/d/Master/Airport_Runway_Detection/vision-based-landing-guidance/output/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-14.21.06-873984/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-14.21.06-873984/eval/epoch_0/results/DINOv2-B-224/MyDataset-test/CYUL_06L_35_test/eval.csv"
    eval_df = pd.read_csv(csv_path, sep=",")

    for i in range(len(eval_df)):
        image_df = eval_df.iloc[i]
        image_name = str(int(image_df["# ind"]) + 1).zfill(4) + ".jpg"
        image_path = os.path.join(input_dir, image_name)
        pred_x = image_df.pred_x
        pred_y = image_df.pred_y
        pred_w = image_df.pred_w
        pred_h = image_df.pred_h

        # pred_x, pred_y, pred_w, pred_h, image_path
        keypoints, crop_data = main_keypoints(pred_x, pred_y, pred_w, pred_h, image_path)

        a = 1