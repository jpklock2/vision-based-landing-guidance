def get_crop_data_from_csv(csvs_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    files = os.listdir(csvs_dir)
    for file in files:

        file_path = os.path.join(csvs_dir, file)
        if os.path.isfile(file_path):
        lard_df = pd.read_csv(file_path, delimiter=';')
        min_percentage = 0.1

        data = []
        for i in lard_df.index:
            filename = os.path.basename(lard_df.loc[i, 'image'])
            width = lard_df.loc[i, 'width']
            height = lard_df.loc[i, 'height']
            xs = [lard_df.loc[i, 'x_A'], lard_df.loc[i, 'x_B'], lard_df.loc[i, 'x_C'], lard_df.loc[i, 'x_D']]
            ys = [lard_df.loc[i, 'y_A'], lard_df.loc[i, 'y_B'], lard_df.loc[i, 'y_C'], lard_df.loc[i, 'y_D']]

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

            data.append([filename, left, upper, abs(right-left), abs(upper-lower)])

        output_file = os.path.join(destination_dir, os.path.basename(file_path))
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

