import argparse
import copy
import pickle
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from pose_estimation_utils import *

file_path = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='estimate pose and distance')
    parser.add_argument('airport_name', help='ICAO airport code')
    parser.add_argument('runways', nargs='+', help='A list of runway codes')
    parser.add_argument('keypoints_file_path',help=('path to detected keypoints file'))
    parser.add_argument('csv_file_path', help='path to groundtruth csv')
    args = parser.parse_args()
    return args

def rotation_vector_to_euler_angles(rotation_vector):
    # Convert rotation vector to rotation matrix
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    r = R.from_matrix(rotation_matrix)
    euler_angles_deg = r.as_euler('xyz', degrees=False)
    order = np.array([1, 0, 2])
    return euler_angles_deg[order]

def normalize_angle(angle):
    normalized_angle = angle % (2 * math.pi)
    if normalized_angle > math.pi:
        normalized_angle -= 2 * math.pi
    elif normalized_angle < -math.pi:
        normalized_angle += 2 * math.pi
    return normalized_angle

def estimate_slant_distance(tvec):
    return np.sqrt(tvec[0, 0]**2 + tvec[1, 0]**2 + tvec[2, 0]**2)

def estimate_pose(corners, airport, runway):
    RUNWAY = (3,2)
    runway_parameters = find_by_airport_runway_number('runway_data.csv', airport, runway)
    runway_width = float(runway_parameters['Width'])
    aspect_ratio = float(runway_parameters['Aspect Ratio'])
    yaw_offset = math.radians(int(runway_parameters['Yaw Offset']))
    n_images = corners.shape[0]
    meters_per_nautic_mile = 1852

    objp = np.zeros((1, RUNWAY[0]*RUNWAY[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:RUNWAY[0]/2:0.5, 0:RUNWAY[1]*aspect_ratio:aspect_ratio].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    no_keypoints_idx = []
    for i in range(n_images):
        if corners[i,0,0,0] == 0:
            no_keypoints_idx.append(i)
        objpoints.append(objp)
        imgpoints.append(corners[i].astype(np.float32))

    camera_matrix_path = os.path.join(file_path, 'camera_calibration/camera_matrix.pkl')
    distortion_path = os.path.join(file_path, 'camera_calibration/distortion.pkl')
    with open(camera_matrix_path, 'rb') as file:
        mtx = pickle.load(file)
    with open(distortion_path, 'rb') as file:
        dist = pickle.load(file)

    rvecs_all, tvecs_all = [], []
    ypr_est = np.zeros((n_images, 3))
    slant_distance_est = np.zeros(n_images)
    
    for i in range(n_images):
        _, rvecs, tvecs = cv2.solvePnP(objp, imgpoints[i], mtx, dist)
        rvecs_all.append(rvecs)
        tvecs_all.append(tvecs)
        ypr_est[i] = rotation_vector_to_euler_angles(rvecs)
        ypr_est[i, 0] += yaw_offset
        ypr_est[i, 2] += np.pi
        ypr_est[i, 2] = -normalize_angle(ypr_est[i, 2])
        slant_distance_est[i] = estimate_slant_distance(tvecs)*runway_width/meters_per_nautic_mile

    return ypr_est, slant_distance_est, no_keypoints_idx

def compute_error(pose_est, slant_distance_est, pose_gt, slant_distance_gt):
    pose_errors = np.degrees(copy.deepcopy(pose_gt-pose_est))
    distance_errors = slant_distance_gt - slant_distance_est
    return pose_errors, distance_errors

def main(airport_name, runways, keypoints_file_path, csv_file_path=None):

    with open(keypoints_file_path, 'rb') as file:
        bbox_coord_kp = pickle.load(file)
    
    estimated_pose, estimated_distance, pose_errors, distance_errors = [], [], [], []
    for runway in runways:
        #Estimates pose based on detected keypoints
        pose_est_kp, slant_distance_est_kp, _ = estimate_pose(bbox_coord_kp, airport_name, runway)
        estimated_pose.append(pose_est_kp)
        estimated_distance.append(slant_distance_est_kp)
        #If the groundtruth csv path is provided, compute the estimation error
        if csv_file_path:
            df, sorted_times = get_dataframe(int(runway), csv_file_path)
            bbox_coord_gt, pose_gt, slant_distance_gt = get_ground_truth(df, sorted_times)
            #Estimates pose based on groundtruth
            #ypr_hat, slant_distance_hat, _ = estimate_pose(bbox_coord_gt, airport_name, runway)
            pose_error, distance_error = compute_error(pose_est_kp, slant_distance_est_kp, pose_gt, slant_distance_gt)
            pose_errors.append(pose_error)
            distance_errors.append(distance_error)

    return estimated_pose, estimated_distance, pose_errors, distance_errors

if __name__ == "__main__":
    args = parse_args()
    estimated_pose, estimated_distance, pose_errors, distance_errors = main(args.airport_name, args.runways, args.keypoints_file_path, args.csv_file_path)
    pose_estimations_path = os.path.join(file_path, 'results/estimated_pose.pkl')
    distance_estimations_path = os.path.join(file_path, 'results/estimated_distance.pkl')
    pose_errors_path = os.path.join(file_path, 'results/pose_errors.pkl')
    distance_errors_path = os.path.join(file_path, 'results/distance_errors.pkl')
    with open(pose_estimations_path, "wb") as f:
        pickle.dump(estimated_pose, f)
    with open(distance_estimations_path, "wb") as f:
        pickle.dump(estimated_distance, f)
    with open(pose_errors_path, "wb") as f:
        pickle.dump(pose_errors, f)
    with open(distance_errors_path, "wb") as f:
        pickle.dump(distance_errors, f)
