import json
import numpy as np
import argparse
import os
import pandas as pd

def main(args):

    if args.shot_iou:
        shot_info_path = args.shot_info_path
        video_ids = [video.split('.')[0] for video in os.listdir(os.path.join(shot_info_path, 'shot_txt'))]

        shot_stats_path = os.path.join(shot_info_path, 'shot_stats')
        shot_txt_path = os.path.join(shot_info_path, 'shot_txt')

        shot_d = {video_id : get_shot_times(shot_stats_path, shot_txt_path, video_id) for video_id in video_ids}

        shot_iou_result = get_iou(args, shot_d)
        print(f"shot을 기준으로 구한 IoU(threshold: {args.threshold}) : {shot_iou_result}")


    if args.scene_iou:
        scene_info_path = args.scene_info_path
        with open(scene_info_path, 'r') as file:
            scene_d = json.load(file)
        
        scene_d = {k:[s['time'] for s in v.values()] for k, v in scene_d.items()}

        scene_iou_result = get_iou(args, scene_d)
        print(f"VSS의 결과로 만들어진 scene을 기준으로 구한 IoU(threshold: {args.threshold}) : {scene_iou_result}")


def get_shot_times(shot_stats_path, shot_txt_path, video_id):
    time_list = []

    stats_path = os.path.join(shot_stats_path, f"{video_id}.csv")
    txt_path = os.path.join(shot_txt_path, f"{video_id}.txt")

    time_info = pd.read_csv(stats_path, skiprows=2, header=None)
    time_info = [0.]+list(time_info.iloc[:, 1])
    time_info = [round(num, 2) for num in time_info]

    frame_info = pd.read_csv(txt_path, header=None, sep='\s+')

    for i in range(len(frame_info)):
        tmp = [time_info[frame_info.iloc[i, 0]], time_info[frame_info.iloc[i, 1]]]
        time_list.append(tmp)

    return time_list


def calculate_iou(interval_1, interval_2):
    start_1, end_1 = interval_1
    start_2, end_2 = interval_2

    # Intersection 구하기
    intersection_start = max(start_1, start_2)
    intersection_end = min(end_1, end_2)
    
    if intersection_end < intersection_start: # intersection이 없는 경우
        return 0.0

    intersection_length = intersection_end - intersection_start

    # Union 구하기
    union_start = min(start_1, start_2)
    union_end = max(end_1, end_2)
    union_length = union_end - union_start

    # Intersection over Union 구하기
    iou = intersection_length / union_length

    return iou


def get_iou(args, interval_dict):

    val_1_path = args.val_1_path
    val_2_path = args.val_2_path

    with open(val_1_path, 'r') as file:
        val_1 = json.load(file)
    with open(val_2_path, 'r') as file:
        val_2 = json.load(file)

    result_list = []

    for k, v in val_1.items():
        for interval_1 in v['timestamps']:
            tmp = []
            for interval_2 in interval_dict[k]:
                tmp.append(calculate_iou(interval_1, interval_2))
                result_list.append(max(tmp))

    for k, v in val_2.items():
        for interval_1 in v['timestamps']:
            tmp = []
            for interval_2 in interval_dict[k]:
                tmp.append(calculate_iou(interval_1, interval_2))
                result_list.append(max(tmp))

    result_list = np.array(result_list)

    result_list = result_list > args.threshold

    return np.mean(result_list)


def get_config():
    parser = argparse.ArgumentParser(description = 'Get IoU for evaluating video scene segmentation')

    parser.add_argument('--shot_iou', action="store_true")
    parser.add_argument('--scene_iou', action="store_true")
    parser.add_argument('--val_1_path', default = "/home/jonghee/Dataset/ActivityNetCaptions/captions/val_1.json", type = str)
    parser.add_argument('--val_2_path', default = "/home/jonghee/Dataset/ActivityNetCaptions/captions/val_2.json", type = str)
    parser.add_argument('--scene_info_path', default = "/home/previ01/changjun/vmr_baseline/scene_segmentation/vss_result/vss_result.json", type = str)
    parser.add_argument('--shot_info_path', default = "/home/previ01/changjun/vmr_baseline/shot_detection/shot_detection_result", type = str)
    parser.add_argument('--threshold', default = 0.5, type = float)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_config()
    main(args)