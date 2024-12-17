# Pose Estimation

The Pose Estimation module can operate independently, using either pre-defined or detected runway corner points.

This module is structured as follows:

```
pose_estimation
├── camera_calibration
│   ├── camera_matrix.pkl
|   └── distortion.pkl
├── pose_estimation.py
├── pose_estimation_utils.py
├── keypoints.pkl
└── runway_data.csv
```

## Camera Calibration

The camera_calibration folder must include two files: camera_matrix.pkl and distortion.pkl. These files contain the camera's intrinsic parameters and distortion coefficients, respectively.

- camera_matrix.pkl: A pickle file storing a NumPy matrix that represents the camera's intrinsic parameters in the following format:

$$
K = \begin{bmatrix}
f_x & 0 & u_0 \\
0 & f_y & v_0 \\
0 & 0 & 1
\end{bmatrix}
$$

where $f_x$ and $f_y$ are the focal lengths in the x and y axis, and $u_0$ and $v_0$ are the camera's principal points in the x and y axis.

- distortion.pkl: A pickle file storing a NumPy array with the camera's distortion coefficients.

## Runway Data

The runway_data.csv file is structured as follows:

| Column Name   | Data Type   | Description    |
|:-------------:|:-------------:|:-------------:|
| Airport    | String | Airport ICAO code |
| Runway    | String | Landing designator   |
| Width    | Float | Runway width   |
| Length    | Float | Runway length   |
| Aspect Ratio    | Float | Runway aspect ratio   |
| Yaw Offset    | Float | Runway azimuth angle    |

It must contain the data of the runway of interest.

## Keypoints Data

For this module to independently estimate the pose, a pickle file containing an array of the runway's corner points for each frame must be provided. The array should have the shape (n, 6, 1, 2), where n is the number of images in the sequence of interest.

Below is an example of a keypoint array for a single image:

```
[[[[x_1	y_1]]]
[[[x_2	y_2]]]
[[[x_3	y_3]]]
[[[x_4	y_4]]]
[[[x_5	y_5]]]
[[[x_6	y_6]]]]
```

where $(x_1, y_1)$ is the lower rightmost corner, $(x_2, y_2)$ is the midway point between the two lower corners, $(x_3, y_3)$ is the lower leftmost corner, $(x_4, y_4)$ is the upper rightmost corner, $(x_5, y_5)$ is the midway point between the two upper corners, and $(x_6, y_6)$ is the upper leftmost corner.

## Ground Truth

To compute the estimation errors (optional), a ground truth file is required. This file must be a CSV with the same structure as the CSV files in the LARD dataset.

## How to Run

