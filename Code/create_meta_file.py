## Code to create the meta file for THUMOS dataset

import json
import os
import math
import collections

meta = {}
action_id_mapping = {}
act_iter = 0
video_id_mapping = {}
video_iter = 0
video_duration = collections.defaultdict(lambda: 1.0)

def time_to_frame_num(i,t):
	return math.floor(float(t)/(video_duration[i]/50))


for filename in os.listdir("./../annotation/"):
	if("Ambiguous" not in filename):
		action_id_mapping[filename.split("_")[0]] = act_iter
		act_iter += 1
		f = open(os.path.join("./../annotation/", filename))
		for line in f:
			video_name, _, start_time, end_time = line.replace("\n","").split(" ")
			if(video_name not in video_id_mapping):
				video_id_mapping[video_name] = video_iter
				video_iter += 1
				meta[video_iter] = {}
				meta[video_iter]['vidName'] = video_name
				#meta[video_iter]['seq'] = [start_frame, end_frame]
				meta[video_iter]['dets'] = [{}]*20
			if(len(meta[video_iter]['dets'][action_id_mapping[filename.split("_")[0]]]) == 0):
				meta[video_iter]['dets'][action_id_mapping[filename.split("_")[0]]] = [[time_to_frame_num(video_iter, start_time), time_to_frame_num(video_iter, end_time)]]
			else:
				meta[video_iter]['dets'][action_id_mapping[filename.split("_")[0]]].append([time_to_frame_num(video_iter, start_time), time_to_frame_num(video_iter, end_time)])

with open('val_meta_file.json', 'w') as outfile:
    json.dump(meta, outfile)