import sys
sys.path.append('vss_evaluation/')
import argparse
import os
import pickle
import json
from iou import get_shot_times

def main(args):
    shot_info_path = args.shot_info_path
    video_ids = [video.split('.')[0] for video in os.listdir(os.path.join(shot_info_path, 'shot_txt'))]

    shot_stats_path = os.path.join(shot_info_path, 'shot_stats')
    shot_txt_path = os.path.join(shot_info_path, 'shot_txt')

    shot_d = {video_id : get_shot_times(shot_stats_path, shot_txt_path, video_id) for video_id in video_ids}

    with open(args.output_path, 'w', encoding='utf-8') as file:
        json.dump(shot_d, file)


def get_config():
    parser = argparse.ArgumentParser(description = 'Get shot information json file with timestamps')

    parser.add_argument('--shot_info_path', default = "/home/previ01/changjun/vmr_baseline/shot_detection/shot_detection_result", type = str)
    parser.add_argument('--output_path', default = "/home/previ01/changjun/vmr_baseline/query_retreival/shot_info_data/shot_info.json", type = str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_config()
    main(args)
