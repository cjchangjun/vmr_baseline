import os
import shutil

keyf_path = './shot_keyf'

for video_id in os.listdir(keyf_path):
    frames = os.listdir(os.path.join(keyf_path, video_id))
    shot_ids = [frame.split('.')[0].split('_')[1] for frame in frames]
    shot_ids = list(set(shot_ids)) 
    for shot_id in shot_ids:
        if f"shot_{shot_id}_img_1.jpg" not in frames:
            shutil.copy(os.path.join(keyf_path, video_id, f"shot_{shot_id}_img_0.jpg"), os.path.join(keyf_path, video_id, f"shot_{shot_id}_img_1.jpg"))
        if f"shot_{shot_id}_img_2.jpg" not in frames:
            shutil.copy(os.path.join(keyf_path, video_id, f"shot_{shot_id}_img_1.jpg"), os.path.join(keyf_path, video_id, f"shot_{shot_id}_img_2.jpg"))