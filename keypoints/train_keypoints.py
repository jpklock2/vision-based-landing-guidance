def train_keypoints(data_dir, train_dir, val_dir, train_weights_dir):
    data_yaml_content = f"""
    train: {train_dir}
    val: {val_dir}

    nc: 1
    names: ['runnaway']

    kpt_shape: [4, 2]
    """

    with open(data_dir, 'w') as file:
        file.write(data_yaml_content)
    
    model = YOLO('yolov8n-pose.yaml')
    model.train(data=data_dir, epochs=100, imgsz=640, batch=16, workers=2, project=train_weights_dir, name='exp', exist_ok=True)
    return model

def val_keypoints(data_dir):
    return model.val(data=data_dir, split='val', save_json=True)

