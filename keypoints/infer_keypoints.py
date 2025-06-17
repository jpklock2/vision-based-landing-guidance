#!pip install ultralytics
import os
import argparse
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from pathlib import Path
from sklearn.linear_model import LinearRegression


def create_cropped_image_path(image_path):
  dir_name, file_name = os.path.split(image_path)
  cropped_dir = os.path.join(dir_name, 'cropped')
  os.makedirs(cropped_dir, exist_ok=True)
  return os.path.join(cropped_dir, file_name)


def calculate_coordinates_new(pred_x, pred_y, pred_w, pred_h, width, height, min_percentage):
  left = int ((pred_x - pred_w) - min_percentage*pred_w)
  left = left if left > 0 else 0

  right = int (pred_x + min_percentage*pred_w)
  right = right if right < width else width

  upper = int ((pred_y - pred_h)- min_percentage*pred_h)
  upper = upper if upper > 0 else 0

  lower = int (pred_y + min_percentage*pred_h)
  lower = lower if lower < height else height
  return left, upper, right, lower


def calculate_coordinates(pred_x, pred_y, pred_w, pred_h, width, height, min_percentage):
      left = int (pred_x - min_percentage*pred_w)
      left = left if left > 0 else 0

      right = int (pred_x + pred_w + min_percentage*pred_w)
      right = right if right < width else width

      upper = int (pred_y - min_percentage*pred_h)
      upper = upper if upper > 0 else 0

      lower = int (pred_y+ pred_h + min_percentage*pred_h)
      lower = lower if lower < height else height
      return left, upper, right, lower


def crop_image(pred_x, pred_y, pred_w, pred_h, image_path, percentage=0.2):
    im = cv2.imread(image_path)
    height, width = im.shape[:2]  # Notice the order: height, width
    left, upper, right, lower = calculate_coordinates(pred_x, pred_y, pred_w, pred_h, width, height, percentage/2)

    # Crop the image using NumPy slicing
    cropped = im[upper:lower, left:right]

    # Resize back to original size
    resized = cv2.resize(cropped, (width, height))

    # Create and save cropped image
    cropped_im_path = create_cropped_image_path(image_path)
    cv2.imwrite(cropped_im_path, resized)

    return cropped_im_path, [image_path, left, upper, abs(right - left), abs(lower - upper)]
def morphological_open_operation(mask, orig_img, iterations=3, kernel_size=3, debug=True):
        def plot():
            h_img, w_img = orig_img.shape[:2]
            resized_mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
    
            # Draw contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 8)
    
            # Show the final result
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Morphological open operation (3 times)")
            plt.show()
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for _ in range(iterations):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
        if debug: plot()
        return mask

def find_center_point(cnt, orig_img, canvas, debug = True):
        def plot():
            # Draw contour
            cv2.drawContours(orig_img, [cnt], -1, (0, 255, 0), 8)
    
            # Show the final result
            plt.figure(figsize=(10, 10))
            cv2.circle(orig_img, tuple([cx, cy]), 15, (255, 0, 0), -1)
            plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Find center points")
            plt.show()
            
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        if debug: plot()
        
        return cx, cy

def find_positioning_points(cnt, center_point, orig_img, num_pixels=15, debug=True):
    def plot():
        plt.figure(figsize=(10, 10))
        cv2.drawContours(orig_img, [cnt], -1, (0, 255, 0), 8)
        for p in center_pts:
            cv2.circle(orig_img, tuple(p), 15, (255, 0, 0), -1)
        cv2.circle(orig_img, tuple([cx, cy]), 15, (0, 255, 0), -1)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Find ps points")
        plt.show()
          
    cx, cy = center_point

    max_y = max([point[0][1] for point in cnt])
    min_y = min([point[0][1] for point in cnt])
    height = int(max_y - min_y)
    prop_y = height/8
    
    # Crie duas linhas horizontais acima e abaixo do centro
    ys = [int(max(cy - i*prop_y, 0)) for i in range(4)]
    ys += [int(min(cy + prop_y, 9999)) for i in range(4)]  # 9999 para não limitar o tamanho
    
    # Obtenha pontos da máscara onde y == y_top ou y_bottom
    center_pts = []
    for y in ys:        
        line = [pt[0] for pt in cnt if pt[0][1] == y]
        # Achar os dois extremos em cada linha (interseções da máscara com a linha)
        if len(line) >= 2:
            line.sort(key=lambda p: p[0])
            pt1, pt2 = line[0], line[-1]
            P = ((pt1[0] + pt2[0]) // 2, y)
        else:
            P = (cx, y)
        center_pts.append(P)
    
    if debug: plot()
    return center_pts

def split_contour_into_sets(cnt, center, ps, orig_img, debug=True):
    def plot():
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]
        for i, quadrant in enumerate(sets):
            for pt in quadrant:
                cv2.circle(orig_img, tuple(pt), 15, colors[i], -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title("Step 4: Corner Points C1–C4  -------------------------------")
        plt.axis("off")
        plt.show()
    cx, cy = center
    
    center_line_model = fit_line(ps)
    a = center_line_model.coef_[0]
    b = center_line_model.intercept_

    def infer_x_from_y(y_value, epsilon=1e-1):
        if abs(a) < epsilon:
            return cx
        return (y_value - b) / a


    sets = [[], [], [], []]  # C1, C2, C3, C4

    for pt in cnt:
        x, y = pt[0]
        predicted_center_x = infer_x_from_y(y)
        if y <= cy:  # Acima de P1
            if x <= predicted_center_x:
                sets[0].append((x, y))  # C1
            else:
                sets[1].append((x, y))  # C2
        #elif y >= y2:  # Abaixo de P2
        else:
            if x <= predicted_center_x:
                sets[2].append((x, y))  # C3
            else:
                sets[3].append((x, y))  # C4

    if any(len(s) == 0 for s in sets):
        sets = [[], [], [], []]
        for pt in cnt:
                x, y = pt[0]
                if y <= cy:  # Acima de P1
                    if x <= cx:
                        sets[0].append((x, y))  # C1
                    else:
                        sets[1].append((x, y))  # C2
                #elif y >= y2:  # Abaixo de P2
                else:
                    if x <= cx:
                        sets[2].append((x, y))  # C3
                    else:
                        sets[3].append((x, y))  # C4
    
    if debug: plot()
    return sets

def find_corner_points(sets, center, orig_img, debug=True):
    def plot():
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]
        for i, pt in enumerate(corners):
            cv2.circle(orig_img, tuple(pt), 15, colors[i], -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title("Step 4: Corner Points C1–C4  -------------------------------")
        plt.axis("off")
        plt.show()
        
    cx, cy = center

    q_1 = np.array(sets[0])
    min_y = np.min(q_1[:, 1])
    candidates = q_1[q_1[:, 1] == min_y]
    top_left = candidates[np.argmin(candidates[:, 0])]

    q_2 = np.array(sets[1])
    min_y = np.min(q_2[:, 1])
    candidates = q_2[q_2[:, 1] == min_y]
    top_right = candidates[np.argmax(candidates[:, 0])]
    
    q_3 = np.array(sets[2])
    min_y = np.max(q_3[:, 1])
    candidates = q_3[q_3[:, 1] == min_y]
    bottom_left = candidates[np.argmin(candidates[:, 0])]
    
    q_4 = np.array(sets[3])
    min_y = np.max(q_4[:, 1])
    candidates = q_4[q_4[:, 1] == min_y]
    bottom_right = candidates[np.argmax(candidates[:, 0])]
    
    corners = [top_left, top_right, bottom_right, bottom_left]
    if debug: plot()
    return corners

def find_corners(cnt, cx, cy, orig_img, debug = True):
    def plot():
        img = orig_img.copy()
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]
        for i, pt in enumerate(corners):
            cv2.circle(img, tuple(pt), 15, colors[i], -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Step 4: Corner Points C1–C4")
        plt.axis("off")
        plt.show()
    
    def plot_quadrants():
        img = orig_img.copy()
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]
        for i, quadrant in enumerate(quadrants):
            for pt in quadrant:
                cv2.circle(img, tuple(pt), 15, colors[i], -1)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Step 4: Corner Points C1–C4  -------------------------------")
        plt.axis("off")
        plt.show()
        
    points = cnt.reshape(-1, 2)
    quadrants = [[] for _ in range(4)]
    for pt in points:
        if pt[0] <= cx and pt[1] <= cy:
            quadrants[0].append(pt)  # Top-left
        elif pt[0] > cx and pt[1] <= cy:
            quadrants[1].append(pt)  # Top-right
        elif pt[0] > cx and pt[1] > cy:
            quadrants[2].append(pt)  # Bottom-right
        else:
            quadrants[3].append(pt)  # Bottom-left

    if debug: plot_quadrants()
    corners = []
    for quad in quadrants:
        if not quad:
            continue
        quad = np.array(quad)
        dists = np.linalg.norm(quad - np.array([cx, cy]), axis=1)
        corners.append(quad[np.argmax(dists)])

    if len(corners) < 4:
        return None
    corners = np.array(corners)

    if debug: plot()
    return corners

def fit_polygon(corners, orig_img, debug=True, title="Fit polygon from corner points"):
        def plot():
            plt.figure(figsize=(10, 10))
            cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 8)
            plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
            plt.show()
        
        h_img, w_img = orig_img.shape[:2]
        poly_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [corners], 255)
        contours, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug: plot()
        return contours

def douglas_peacker(cnt, H, W, title, debug=True):
        epsilon = 0.001 * np.log(cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return approx

def douglas_peacker(cnt, H, W, title, debug=True):
    arc_len = cv2.arcLength(cnt, True)
    epsilon = 0.001 * arc_len
    max_epsilon = 0.2 * arc_len  # limit to avoid over-simplification
    step = 0.001 * arc_len

    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Iteratively increase epsilon until we get exactly 4 points
    while len(approx) != 4 and epsilon < max_epsilon:
        epsilon += step
        approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 4:
        return None
    if debug:
        print(f"[{title}] Final epsilon: {epsilon:.4f}, Points: {len(approx)}")

    return approx

def point_line_distance(point, line_start, line_end):
    """Calcula a distância de um ponto até uma linha definida por dois pontos."""
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point - line_start)
    proj = np.dot(point_vec, line_vec) / line_len
    proj_point = line_start + (proj / line_len) * line_vec
    return np.linalg.norm(point - proj_point)

def plot_threshold_sides(orig_img, left_pts, bottom_pts, right_pts, top_pts, point_radius=15):
    """Plota os pontos de cada lado com cores diferentes no canvas."""
    orig_img_copy = orig_img.copy()

    for pt in left_pts:
        cv2.circle(orig_img_copy, tuple(pt), point_radius, (255, 0, 0), -1)  # Azul
    for pt in bottom_pts:
        cv2.circle(orig_img_copy, tuple(pt), point_radius, (0, 255, 0), -1)  # Verde
    for pt in right_pts:
        cv2.circle(orig_img_copy, tuple(pt), point_radius, (0, 0, 255), -1)  # Vermelho
    for pt in top_pts:
        cv2.circle(orig_img_copy, tuple(pt), point_radius, (128, 0, 128), -1)  # Roxo

    # Exibe com Matplotlib (converte BGR para RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img_copy[..., ::-1])
    plt.title("Pontos divididos por lado")
    plt.axis("off")
    plt.show()

def fit_line(points):
    points = np.array(points).reshape(-1, 2)
    X = points[:, 0].reshape(-1, 1)  # coordenadas x
    y = points[:, 1]                # coordenadas y
    model = LinearRegression().fit(X, y)
    return model

def identify_side(top_left, top_right, bottom_right, bottom_left, contour, orig_image, debug=True):
    left_side = (top_left, bottom_left)
    bottom_side = (bottom_left, bottom_right)
    right_side = (bottom_right, top_right)
    top_side = (top_right, top_left)

    actual_left_side = []
    actual_bottom_side = []
    actual_right_side = []
    actual_top_side = []

    for pt in contour.reshape(-1, 2):
        d_left = point_line_distance(pt, *left_side)
        d_bottom = point_line_distance(pt, *bottom_side)
        d_right = point_line_distance(pt, *right_side)
        d_top = point_line_distance(pt, *top_side)

        dists = [d_left, d_bottom, d_right, d_top]
        min_index = np.argmin(dists)

        if min_index == 0:
            actual_left_side.append(pt)
        elif min_index == 1:
            actual_bottom_side.append(pt)
        elif min_index == 2:
            actual_right_side.append(pt)
        else:
            actual_top_side.append(pt)

    left_pts, bottom_pts, right_pts, top_pts = (
        np.array(actual_left_side),
        np.array(actual_bottom_side),
        np.array(actual_right_side),
        np.array(actual_top_side)
    )
    if debug: plot_threshold_sides(orig_image, left_pts, bottom_pts, right_pts, top_pts)
    return left_pts, bottom_pts, right_pts, top_pts

def get_new_corner_points(model, x_min, x_max):
    # Cria dois pontos extremos na horizontal
    x_vals = np.array([x_min, x_max]).reshape(-1, 1)
    y_vals = model.predict(x_vals).astype(np.int32)
    pt1 = (int(x_vals[0][0]), int(y_vals[0]))
    pt2 = (int(x_vals[1][0]), int(y_vals[1]))
    return np.array([pt1, pt2], dtype=np.int32)

def get_unique_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def post_process_runway(mask, original_image, debug=True):
    h_img, w_img = original_image.shape[:2]

    mask_np = (mask.cpu().numpy() * 255).astype('uint8')
    mask_np = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    H, W = mask_np.shape
    
    mask = morphological_open_operation(mask_np, original_image.copy(), debug=debug)
    canvas_base = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    
    cx, cy = find_center_point(cnt, original_image.copy(), canvas_base.copy(), debug=debug)
    
    ps = find_positioning_points(cnt, (cx, cy), original_image.copy(), debug=debug)
    sets = split_contour_into_sets(cnt, (cx, cy), ps, original_image.copy(), debug=debug)
    corners = find_corner_points(sets, (cx, cy), original_image.copy(), debug=debug)

    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]

    _, bottom_pts, _, _ = identify_side(top_left, top_right, bottom_right, bottom_left, cnt, original_image.copy(), debug=debug)
    bottom_model = fit_line(bottom_pts)
    
    x_min, x_max = np.min(bottom_pts, axis=0)[0], np.max(bottom_pts, axis=0)[0]
    bottom_line_pts = get_new_corner_points(bottom_model, x_min, x_max)
    new_left_bottom, new_right_bottom = bottom_line_pts
    
    new_corners = np.array([top_left, top_right, new_right_bottom, new_left_bottom], dtype=np.int32)

    fit_polygon(new_corners, original_image.copy(), debug=debug, title="post_process_runway")
    return new_corners

def post_process_threshold(mask, original_image, debug=True):
    h_img, w_img = original_image.shape[:2]
    mask_np = (mask.cpu().numpy() * 255).astype('uint8')
    mask_np = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    H, W = mask_np.shape
    # Step 1: Morphological open operation (3 times)
    mask = morphological_open_operation(mask_np, original_image.copy(), debug=debug)
    canvas_base = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # Step 2: Find contours}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    cx, cy = find_center_point(cnt, original_image.copy(), canvas_base.copy(), debug=debug)
    
    # Step 3: Find corner points C1–C4 (farthest from O in each quadrant)
    corners = find_corners(cnt, cx, cy, original_image.copy(), debug=debug)
    
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]

    # Step 4: find sides
    _, bottom_pts, _, top_pts = identify_side(top_left, top_right, bottom_right, bottom_left, cnt, original_image.copy(), debug=debug)

    # Step 5: Apply linear regression
    bottom_model = fit_line(bottom_pts)
    top_model = fit_line(top_pts)

    # Step 6: Predict new corners
    x_min, x_max = np.min(bottom_pts, axis=0)[0], np.max(bottom_pts, axis=0)[0]
    bottom_line_pts = get_new_corner_points(bottom_model, x_min, x_max)
    new_left_bottom, new_right_bottom = bottom_line_pts
    
    x_min, x_max = np.min(top_pts, axis=0)[0], np.max(top_pts, axis=0)[0]
    top_line_pts = get_new_corner_points(top_model, x_min, x_max)
    new_left_top, new_right_top = top_line_pts
    
    new_corners = np.array([new_left_top, new_right_top, new_right_bottom, new_left_bottom], dtype=np.int32)

    fit_polygon(new_corners, original_image.copy(), debug=debug, title="post_process_threshold")
   
    return new_corners

def post_process_aiming_points(mask, original_image, debug=True):
    h_img, w_img = original_image.shape[:2]
    mask_np = (mask.cpu().numpy() * 255).astype('uint8')
    mask_np = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    H, W = mask_np.shape
    # Step 1: Morphological open operation (3 times)
    mask = morphological_open_operation(mask_np, original_image.copy(), debug=debug)
    canvas_base = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # Step 2: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return douglas_peacker(cnt, H, W, "Aiming Point", debug=debug)

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def fix_mask_angles(corners, angles):
    top_left, top_right, _, _ = corners
    θ_esq, θ_dir = angles

    θ_aux = θ_esq - 90
    height = np.linalg.norm(corners[3] - corners[0]) /np.cos(np.radians(θ_aux))

    θ_esq_rad = np.radians(θ_esq)
    θ_dir_rad = np.radians(θ_dir)
    
    left_vec = height * np.array([np.cos(θ_esq_rad), np.sin(θ_esq_rad)])
    right_vec = height * np.array([-np.cos(θ_dir_rad), np.sin(θ_dir_rad)])

    # Calcula novas quinas inferiores
    new_bottom_left = top_left + left_vec
    new_bottom_right = top_right + right_vec

    new_corners = np.array([top_left, top_right, new_bottom_right, new_bottom_left], dtype=np.int32)
    return new_corners

def plot_original_masks(result, orig_img):
    masks = result.masks.data.cpu().numpy()  # shape: (N, h, w)
    num_masks, h_mask, w_mask = masks.shape
    h_img, w_img = orig_img.shape[:2]

    for i in range(num_masks):
        # Resize mask to match original image size
        resized_mask = cv2.resize(masks[i], (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

        # Draw contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 2)

    # Show the final result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Resized Masks Overlay")
    plt.show()

def find_coeficients(p_0, p_1):
    x_0, y_0 = p_0[0], p_0[1]
    x_1, y_1 = p_1[0], p_1[1]
    a = (y_1 - y_0)/(x_1 - x_0)
    b = y_0 - a*x_0
    return a, b

def safe_find_coeficients(p1, p2):
        if p1[0] == p2[0]:
            return 0, 0
        a, b = find_coeficients(p1, p2)
        if np.isnan(a) or np.isinf(a):
            return 0, 0  # assume horizontal
        return a, b

def project_runway(runway_corners, threshold_corners):
     # Calculate slopes safely
    #print(f"Runway top coordinates: {runway_corners[0]} -> {runway_corners[1]}")
    runway_top_a, _ = safe_find_coeficients(runway_corners[0], runway_corners[1])
    #print(f"Runway bottom coordinates: {runway_corners[3]} -> {runway_corners[2]}")
    runway_bottom_a, _ = safe_find_coeficients(runway_corners[3], runway_corners[2])
    #print(f"Threshold top coordinates: {threshold_corners[0]} -> {threshold_corners[1]}")
    threshold_top_a, _ = safe_find_coeficients(threshold_corners[0], threshold_corners[1])
    #print(f"Threshold bottom coordinates: {threshold_corners[3]} -> {threshold_corners[2]}")
    threshold_bottom_a, _ = safe_find_coeficients(threshold_corners[3], threshold_corners[2])

    #print("runway_top_a", runway_top_a)
    #print("runway_bottom_a", runway_bottom_a)
    #print("threshold_top_a", threshold_top_a)
    #print("threshold_bottom_a", threshold_bottom_a)
    
    # Filter out outliers and average
    slopes = [runway_top_a, runway_bottom_a, threshold_top_a, threshold_bottom_a]
    slopes = [s for s in slopes if not np.isnan(s) and not np.isinf(s) and s >= 0]
    if not slopes:
        a = 0
    else:
        mean = np.mean(slopes)
        std = np.std(slopes)
        filtered = [s for s in slopes if abs(s - mean) < 2 * std]
        a = np.mean(filtered) if filtered else mean
    #print("a", a)
    # Construct new bottom corners for runway
    #bottom_y = max(runway_corners[2][1], runway_corners[3][1], threshold_corners[2][1], threshold_corners[3][1])
    bottom_left_y = max(runway_corners[3][1], threshold_corners[3][1])
    bottom_right_y = max(runway_corners[2][1], threshold_corners[2][1])
    left_x = min(runway_corners[3][0], threshold_corners[3][0])
    right_x = max(runway_corners[2][0], threshold_corners[2][0])

    bottom_left_y = a * left_x + bottom_left_y
    bottom_right_y = a * right_x + bottom_right_y

    return np.array([
        runway_corners[0],
        runway_corners[1],
        [right_x, bottom_right_y],
        [left_x, bottom_left_y]
    ], dtype=np.int32)

def is_on_edge(coords, width):
    left_threshold = 0.1 * width
    right_threshold = 0.9 * width

    for coord in coords:
        if coord[0] <= left_threshold or coord[0] >= right_threshold:
            return True
    return False

def unites_runway_and_threshold_masks(runway_corners, threshold_corners, orig_img, debug=True):
    # Ensure order and type
    runway_corners = np.array(runway_corners, dtype=np.int32)
    fit_polygon(runway_corners, orig_img.copy(), debug=debug)
    threshold_corners = np.array(threshold_corners, dtype=np.int32)
    if len(runway_corners) != 4 or len(threshold_corners) != 4:
        raise ValueError("Both masks must have exactly 4 corner points")

    new_runway_corners = runway_corners
    new_runway_contour = fit_polygon(new_runway_corners, orig_img.copy(), debug=debug)
    if not is_on_edge(runway_corners, orig_img.shape[:2][1]):
        new_runway_corners = project_runway(runway_corners, threshold_corners)
    
    new_runway_contour = fit_polygon(new_runway_corners, orig_img.copy(), debug=debug)
    new_runway_contour = max(new_runway_contour, key=cv2.contourArea)

    # Fix threshold corners by angles
    θ_esq = angle_between_vectors(new_runway_corners[1] - new_runway_corners[0],
                                  new_runway_corners[3] - new_runway_corners[0])
    θ_dir = angle_between_vectors(new_runway_corners[0] - new_runway_corners[1],
                                  new_runway_corners[2] - new_runway_corners[1])
        
    new_threshold_corners = fix_mask_angles(threshold_corners, [θ_esq, θ_dir])
    fit_polygon(new_threshold_corners, orig_img.copy(), debug=debug, title="Threshold after fix max angles")
    
    # Final unified mask if threshold size is close to threshold size
    if (new_threshold_corners[2][0] -  new_threshold_corners[3][0]) > 0.8*(new_runway_corners[2][0] -  new_runway_corners[3][0]):
        new_runway_corners = np.array([
            runway_corners[0],
            runway_corners[1],
            new_threshold_corners[2],
            new_threshold_corners[3]
        ], dtype=np.int32)

    fit_polygon(new_runway_corners, orig_img.copy(), debug=debug, title="Final runway")
    return new_runway_corners, new_threshold_corners

def select_best_aiming_points(runway_corners, aiming_points, orig_img, debug=True):
    if debug: plot_all_masks(runway_corners, None, aiming_points, orig_img.copy())
    # Cálculo da média das coordenadas x dos pontos superiores da pista
    mean_x = np.mean([runway_corners[0][0], runway_corners[1][0]])

    left_aiming_points = []
    right_aiming_points = []

    # Classifica os aiming points à esquerda ou à direita com base na coordenada x
    for aiming_point in aiming_points:
        rightest_coordinate = max(aiming_point[1][0][0], aiming_point[2][0][0])  # x de dois pontos superiores
        if rightest_coordinate < mean_x:
            left_aiming_points.append(aiming_point)
        else:
            right_aiming_points.append(aiming_point)

    # Função auxiliar: calcula média do y dos dois pontos superiores (0 e 1)
    def top_avg_y(pts):
        return np.mean([pts[0][0][1], pts[1][0][1]])

    left_aiming_point= None
    right_aiming_point = None
    # Seleciona o ponto de mira mais ao topo (menor y) entre os da esquerda
    if left_aiming_points:
        left_aiming_point = min(left_aiming_points, key=lambda pts: top_avg_y(pts))

    # Seleciona o ponto de mira mais ao topo (menor y) entre os da direita
    if right_aiming_points:
        right_aiming_point= min(right_aiming_points, key=lambda pts: top_avg_y(pts))
        
    return add_missing_aiming_point(runway_corners, left_aiming_point, right_aiming_point, orig_img, debug=debug)

def plot_all_masks(runway_corners, threshold_corners, aiming_points, orig_img):
    img = orig_img.copy()

    if runway_corners is not None:
        runway_contour =  fit_polygon(runway_corners, orig_img.copy(), debug=False)
        cv2.drawContours(img, runway_contour, -1, (0, 255, 0), 8)

    if threshold_corners is not None:
            threshold_contour =  fit_polygon(threshold_corners, orig_img.copy(), debug=False)
            cv2.drawContours(img, threshold_contour, -1, (0, 255, 0), 8)

    # Plot aiming points (Douglas-Peucker results)
    if aiming_points is not None:
        for aiming_point in aiming_points:
            aiming_contour =  fit_polygon(aiming_point, orig_img.copy(), debug=False)
            cv2.drawContours(img, aiming_contour, -1, (0, 255, 0), 8)

    # Display result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("All Masks: Runway, Threshold, Aiming Points")
    plt.axis('off')
    plt.show()

def normalize_contour(contour, img_width, img_height):
    return [(pt[0] / img_width, pt[1] / img_height) for pt in contour]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def assign_thresholds(runways, thresholds, assignments):
    for threshold in thresholds:
        tl_thresh, tr_thresh = threshold[0], threshold[1]
        best_runway_idx = None
        min_avg_dist = float('inf')

        for r_idx, runway in enumerate(runways):
            bl_runway, br_runway = runway[3], runway[2]
            avg_dist = (euclidean(tl_thresh, bl_runway) + euclidean(tr_thresh, br_runway)) / 2

            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_runway_idx = r_idx

        assignments[best_runway_idx]["threshold"] = threshold
    return assignments

def assign_aps(runways, aiming_points, assignments):
    assigned_aps = set()

    for ap in aiming_points:
        best_runway_idx = None
        max_inside = -1

        for r_idx, runway in enumerate(runways):
            inside_count = 0

            for pt in ap:
                x, y = pt[0]  # Extract (x, y) from [[x, y]]
                x_min = min(p[0] for p in runway)
                x_max = max(p[0] for p in runway)
                y_min = min(p[1] for p in runway)
                y_max = max(p[1] for p in runway)

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    inside_count += 1

            if inside_count > max_inside:
                max_inside = inside_count
                best_runway_idx = r_idx

        ap_key = tuple(map(tuple, ap.reshape(-1, 2)))  # Unique hashable key

        if best_runway_idx is not None and ap_key not in assigned_aps:
            if "aiming_points" not in assignments[best_runway_idx]:
                assignments[best_runway_idx]["aiming_points"] = []

            assignments[best_runway_idx]["aiming_points"].append(ap)
            assigned_aps.add(ap_key)
            
    for a in assignments:
        if "aiming_points" not in a:
            a["aiming_points"] = []
    return assignments

def assign_thresholds_and_aps(runways, thresholds, aiming_points):
    assignments = []

    # Initialize empty assignments
    for runway in runways:
        assignments.append({
            "runway": runway,
            "threshold": None,
            "aiming_point_1": None,
            "aiming_point_2": None
        })
    assignments = assign_thresholds(runways, thresholds, assignments)
    assignments = assign_aps(runways, aiming_points, assignments)
    return assignments   

def generate_annotations(groups_annotations, orig_img, destination_path):
    H, W = orig_img.shape[:2]
    os.makedirs(destination_path, exist_ok=True)

    output_txt = os.path.join(destination_path, "contours.txt")
    output_img = os.path.join(destination_path, "annotated_image.jpg")

    img = np.zeros((H, W, 3), dtype=np.uint8)

    with open(output_txt, "w") as f:
        for runway_corners, threshold_corners, aiming_points in groups_annotations:
            if runway_corners is not None:
                normalized = normalize_contour(runway_corners, W, H)
                f.write("0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized]) + "\n")

            if threshold_corners is not None:
                normalized = normalize_contour(threshold_corners, W, H)
                f.write("1 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized]) + "\n")

            if aiming_points is not None:
                for aiming_point in aiming_points:
                    aiming_point = aiming_point.reshape(-1, 2)
                    normalized = normalize_contour(aiming_point, W, H)
                    f.write("2 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized]) + "\n")

    cv2.imwrite(output_img, img)

def add_missing_aiming_point(runway, left_aiming_point, right_aiming_point, orig_img, debug=True):
    def find_x_coordinate(y, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 == x2:
            return x1  # vertical line: x is constant
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        if m == 0:
            return None  # horizontal line: x cannot be solved uniquely
        x = (y - b) / m
        return int(x)

    if left_aiming_point is not None and right_aiming_point is not None:
        return [left_aiming_point, right_aiming_point]
    new_left_aiming_point = left_aiming_point
    new_right_aiming_point = right_aiming_point
    
    if left_aiming_point is not None:
        left_aiming_point = left_aiming_point.reshape(-1, 2)
        top_left, top_right, bottom_right, bottom_left = left_aiming_point[0], left_aiming_point[1], left_aiming_point[2], left_aiming_point[3]
    
        r_top_left, r_top_right, r_bottom_right, r_bottom_left  = runway[0], runway[1], runway[2], runway[3]
        
        top_left_runway_corresponding_coordinate = find_x_coordinate(y=top_left[1], pt1=r_top_left, pt2=r_bottom_left)
        bottom_left_runway_corresponding_coordinate = find_x_coordinate(y=bottom_left[1], pt1=r_top_left, pt2=r_bottom_left)
        
        top_ap_width = top_right[0] - top_left[0]
        bottom_ap_width =  bottom_right[0] - bottom_left[0]

        top_dist_from_margin = top_left[0] - top_left_runway_corresponding_coordinate
        bottom_dist_from_margin = bottom_left[0] - bottom_left_runway_corresponding_coordinate

        top_right_runway_corresponding_coordinate = find_x_coordinate(y=top_right[1], pt1=r_top_right, pt2=r_bottom_right)
        bottom_right_runway_corresponding_coordinate = find_x_coordinate(y=bottom_right[1], pt1=r_top_right, pt2=r_bottom_right)
        
        new_top_right_x = top_right_runway_corresponding_coordinate - top_dist_from_margin     
        new_bottom_right_x = bottom_right_runway_corresponding_coordinate - bottom_dist_from_margin     
        new_top_left_x = new_top_right_x - top_ap_width
        new_bottom_left_x = new_bottom_right_x - bottom_ap_width

        a, _ = safe_find_coeficients(r_top_left, r_top_right)
        if a is None:
            new_right_aiming_point = None
        else:
            new_top_left_y = a * new_top_left_x + top_left[1]
            new_top_right_y = a * new_top_right_x + top_right[1]
            new_bottom_right_y = a * new_bottom_right_x + bottom_right[1]
            new_bottom_left_y = a * new_bottom_left_x + bottom_left[1]
            
            new_right_aiming_point = np.array([
                [new_top_left_x, new_top_left_y],
                [new_top_right_x, new_top_right_y],
                [new_bottom_right_x, new_bottom_right_y],
                [new_bottom_left_x, new_bottom_left_y],
            ], dtype=np.int32)
        
    elif right_aiming_point is not None:
        right_aiming_point = right_aiming_point.reshape(-1, 2)
        top_left, top_right, bottom_right, bottom_left  = right_aiming_point[0], right_aiming_point[1], right_aiming_point[2], right_aiming_point[3]

        r_top_left, r_top_right, r_bottom_right, r_bottom_left  = runway[0], runway[1], runway[2], runway[3]
    
        top_right_runway_corresponding_coordinate = find_x_coordinate(y=top_right[1], pt1=r_top_right, pt2=r_bottom_right)
        bottom_right_runway_corresponding_coordinate = find_x_coordinate(y=bottom_right[1], pt1=r_top_right, pt2=r_bottom_right)
        
        top_ap_width = top_right[0] - top_left[0]
        bottom_ap_width =  bottom_right[0] - bottom_left[0]

        top_dist_from_margin = top_right_runway_corresponding_coordinate - top_right[0]
        bottom_dist_from_margin = bottom_right_runway_corresponding_coordinate - bottom_right[0]

        top_left_runway_corresponding_coordinate = find_x_coordinate(y=top_left[1], pt1=r_top_left, pt2=r_bottom_left)
        bottom_left_runway_corresponding_coordinate = find_x_coordinate(y=bottom_left[1], pt1=r_top_left, pt2=r_bottom_left)
        
        new_top_left_x = top_left_runway_corresponding_coordinate + top_dist_from_margin     
        new_bottom_left_x = bottom_left_runway_corresponding_coordinate + bottom_dist_from_margin     
        new_top_right_x = new_top_left_x + top_ap_width
        new_bottom_right_x = new_bottom_left_x + bottom_ap_width

        a, _ = safe_find_coeficients(r_top_left, r_top_right)
        if a is None:
            new_left_aiming_point = None
        else:
            new_top_left_y = a * new_top_left_x + top_left[1]
            new_top_right_y = a * new_top_right_x + top_right[1]
            new_bottom_right_y = a * new_bottom_right_x + bottom_right[1]
            new_bottom_left_y = a * new_bottom_left_x + bottom_left[1]
            
            new_left_aiming_point = np.array([
                [new_top_left_x, new_top_left_y],
                [new_top_right_x, new_top_right_y],
                [new_bottom_right_x, new_bottom_right_y],
                [new_bottom_left_x, new_bottom_left_y],
            ], dtype=np.int32)

    aiming_points = []
    if new_left_aiming_point is not None:
        aiming_points.append(new_left_aiming_point)
    if new_right_aiming_point is not None:
        aiming_points.append(new_right_aiming_point)
    return aiming_points

def post_process_masks(result, image_path, destination_path, iou_threshold=0.6, debug=True):
    if result.masks is None or not hasattr(result.masks, 'data') or result.masks.data is None:
        return

    orig_img = cv2.imread(image_path)
    if debug: plot_original_masks(result, orig_img.copy())

    runways = []
    thresholds = []
    aiming_points = []
    for i, mask in enumerate(result.masks.data):
        class_id = int(result.boxes.cls[i].item())
        if class_id == 0:
            runways.append(post_process_runway(mask, orig_img.copy(), debug=debug))
        elif class_id == 1: 
            thresholds.append(post_process_threshold(mask, orig_img.copy(), debug=debug))
        elif class_id == 2:
            aiming_point = post_process_aiming_points(mask, orig_img.copy(), debug=debug)
            if aiming_point is not None:
                aiming_points.append(aiming_point)
                
    if runways:
        groups = assign_thresholds_and_aps(runways, thresholds, aiming_points)
        
        all_annotations = []
        for group in groups:
            runway_corners = group["runway"]
            threshold_corners = group["threshold"]
            aiming_pts = group["aiming_points"]
        
            if (runway_corners is not None) and (threshold_corners is not None):
                runway_corners, threshold_corners = unites_runway_and_threshold_masks(runway_corners, threshold_corners, orig_img.copy(), debug=debug)
            if (runway_corners is not None) and (aiming_pts is not None):
                aiming_pts = select_best_aiming_points(runway_corners, aiming_pts, orig_img.copy(), debug=debug)
        
            all_annotations.append((runway_corners, threshold_corners, aiming_pts))
            if debug: plot_all_masks(runway_corners, threshold_corners, aiming_pts, orig_img.copy())
        return runway_corners
        #generate_annotations(all_annotations, orig_img.copy(), destination_path)
    else:
        threshold_corners = thresholds[0] if len(thresholds) > 0 else None
        #generate_annotations(None, threshold_corners, aiming_points, orig_img.copy(), destination_path)
        if debug: plot_all_masks(None, threshold_corners, aiming_points, orig_img.copy())
        return runway_corners
        
def main_keypoints(pred_x, pred_y, pred_w, pred_h, image_path):
  cropped_im_path, crop_data = crop_image(pred_x, pred_y, pred_w, pred_h, image_path)
  model_path = os.path.join("models", "keypoints", "model.pt")
  model = YOLO(model_path)
  keypoints = post_process_masks(results[0], cropped_im_path, "", debug=True)
  # keypoints = results[0].keypoints.xy

  return keypoints, [crop_data]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the image around the tracking detection and predict keypoints")
    parser.add_argument("pred_x", type=int, help="Predicted x coordinate")
    parser.add_argument("pred_y", type=int, help="Predicted y coordinate")
    parser.add_argument("pred_w", type=int, help="Predicted width")
    parser.add_argument("pred_h", type=int, help="Predicted height")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    keypoints, crop_data = main_keypoints(args.pred_x, args.pred_y, args.pred_w, args.pred_h, args.image_path)
