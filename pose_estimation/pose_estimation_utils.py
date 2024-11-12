import os
import csv
import pickle
import pandas as pd
import numpy as np

dataset_path = os.path.dirname(os.path.abspath(__file__))

def find_by_airport_runway_number(file_name, airport, runway):
    csv_path = os.path.join(dataset_path, file_name)
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)   
        for row in reader:
            if row['Airport'] == airport and row['Runway'] == runway:
                return row           
    return None

def get_dataframe(runway, csv_file_name, airport=None):
    #dataset_path  = os.getcwd()
    csv_path = os.path.join(dataset_path, csv_file_name)
    
    df = pd.read_csv(csv_path, sep=';', parse_dates=["time"])
    if airport:
        df = df[(df.airport == airport)&(df.runway == runway)]
    elif runway:
        df = df[(df.runway == runway)]    
    df = df.sort_values("slant_distance", ascending=False)
    sorted_times = list(df.index)
    return df, sorted_times

def interp(x1, x2):
    return (x1+x2)/2

def switch(a, b):
    return b, a

def get_ground_truth(df, sorted_times):
    n_images = len(sorted_times)
    cnt = 1
    bbox_coord = np.zeros((n_images, 6, 1, 2))
    ypr = np.zeros((n_images,3))
    slant_distance = np.zeros(n_images)
    for idx in sorted_times[:n_images]:
        
        ypr[cnt-1, 0] = df.yaw.loc[idx]
        ypr[cnt-1, 1] = df.pitch.loc[idx]
        ypr[cnt-1, 2] = df.roll.loc[idx]

        slant_distance[cnt-1] = df.slant_distance.loc[idx]

        x_1 = df.x_C.loc[idx]
        x_2 = df.x_D.loc[idx]
        x_3 = df.x_A.loc[idx]
        x_4 = df.x_B.loc[idx]
        y_1 = df.y_C.loc[idx]
        y_2 = df.y_D.loc[idx]
        y_3 = df.y_A.loc[idx]
        y_4 = df.y_B.loc[idx]

        if x_1 > x_2:       
            x_1, x_2 = switch(x_1, x_2)
            y_1, y_2 = switch(y_1, y_2)

        if x_3 > x_4:       
            x_3, x_4 = switch(x_3, x_4)
            y_3, y_4 = switch(y_3, y_4)

        bbox_coord[cnt-1, 2, 0, 0] = x_1
        bbox_coord[cnt-1, 2, 0, 1] = y_1
        bbox_coord[cnt-1, 1, 0, 0] = int(interp(x_1, x_2))
        bbox_coord[cnt-1, 1, 0, 1] = int(interp(y_1, y_2))
        bbox_coord[cnt-1, 0, 0, 0] = x_2
        bbox_coord[cnt-1, 0, 0, 1] = y_2
    
        bbox_coord[cnt-1, 5, 0, 0] = x_3
        bbox_coord[cnt-1, 5, 0, 1] = y_3
        bbox_coord[cnt-1, 4, 0, 0] = int(interp(x_3, x_4))
        bbox_coord[cnt-1, 4, 0, 1] = int(interp(y_3, y_4))
        bbox_coord[cnt-1, 3, 0, 0] = x_4
        bbox_coord[cnt-1, 3, 0, 1] = y_4
    
        cnt += 1

    ypr = np.radians(ypr)

    return bbox_coord, ypr, slant_distance

def format_keypoints(detected_keypoints, crop_df_path):
    output_path = "bbox_coord_cyyz_seq.pkl"
    original_img_width = 2448
    original_img_height = 2648

    crop_df = pd.read_csv(crop_df_path, delimiter=',')
    n_images = len(crop_df)
    bbox_coord = np.zeros((n_images, 6, 1, 2))

    for i in range(n_images):
        kp = detected_keypoints[i][0].keypoints.data.cpu()
        row_data = crop_df.iloc[i]
        left, upper, width, height = row_data['left'], row_data['upper'], row_data['width'], row_data['height']

        # Return keypoints to original image dimentions
        x_1 = int(kp[0, 0, 0] / original_img_width * width + left)
        x_2 = int(kp[0, 1, 0] / original_img_width * width + left)
        x_3 = int(kp[0, 2, 0] / original_img_width * width + left)
        x_4 = int(kp[0, 3, 0] / original_img_width * width + left)
        y_1 = int(kp[0, 0, 1] / original_img_height * height + upper)
        y_2 = int(kp[0, 1, 1] / original_img_height * height + upper)
        y_3 = int(kp[0, 2, 1] / original_img_height * height + upper)
        y_4 = int(kp[0, 3, 1] / original_img_height * height + upper)

        x_list = np.array([x_1, x_2, x_3, x_4])
        y_list = np.array([y_1, y_2, y_3, y_4])

        #Standardizing keypoint order
        sorted_idx = np.argsort(y_list)

        if x_list[sorted_idx[-1]] < x_list[sorted_idx[-2]]:
            x_1 = x_list[sorted_idx[-1]]
            y_1 = y_list[sorted_idx[-1]]
            x_2 = x_list[sorted_idx[-2]]
            y_2 = y_list[sorted_idx[-2]]
        else:
            x_1 = x_list[sorted_idx[-2]]
            y_1 = y_list[sorted_idx[-2]]
            x_2 = x_list[sorted_idx[-1]]
            y_2 = y_list[sorted_idx[-1]]

        if x_list[sorted_idx[1]] < x_list[sorted_idx[0]]:
            x_3 = x_list[sorted_idx[1]]
            y_3 = y_list[sorted_idx[1]]
            x_4 = x_list[sorted_idx[0]]
            y_4 = y_list[sorted_idx[0]]
        else:
            x_3 = x_list[sorted_idx[0]]
            y_3 = y_list[sorted_idx[0]]
            x_4 = x_list[sorted_idx[1]]
            y_4 = y_list[sorted_idx[1]]

        #Formating keypoints to pose estimation input format
        bbox_coord[i, 2, 0, 0] = x_1
        bbox_coord[i, 2, 0, 1] = y_1
        bbox_coord[i, 1, 0, 0] = int(interp(x_1, x_2))
        bbox_coord[i, 1, 0, 1] = int(interp(y_1, y_2))
        bbox_coord[i, 0, 0, 0] = x_2
        bbox_coord[i, 0, 0, 1] = y_2

        bbox_coord[i, 5, 0, 0] = x_3
        bbox_coord[i, 5, 0, 1] = y_3
        bbox_coord[i, 4, 0, 0] = int(interp(x_3, x_4))
        bbox_coord[i, 4, 0, 1] = int(interp(y_3, y_4))
        bbox_coord[i, 3, 0, 0] = x_4
        bbox_coord[i, 3, 0, 1] = y_4

    with open(output_path, "wb") as file:
        pickle.dump(bbox_coord, file)