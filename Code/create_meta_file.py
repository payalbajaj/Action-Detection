## Code to create the meta file for THUMOS dataset

import json
import os
import math
import collections
import pickle

fileObject = open("./../train_videos.pkl",'r')  
train_videos = pickle.load(fileObject)  
train_list = train_videos.keys()

fileObject = open("./../validation_videos.pkl",'r')  
val_videos = pickle.load(fileObject)  
val_list = val_videos.keys()

# fileObject = open("test_videos.pkl",'r')  
# test_videos = pickle.load(fileObject)  
# test_list = test_videos.keys()
test_list = []

meta = [{}, {}, {}]
action_id_mapping = {'HighJump': 40, 'ThrowDiscus': 93, 'PoleVault': 68, 'VolleyballSpiking': 97, 'BasketballDunk': 9, 'CleanAndJerk': 21, 'Billiards': 12, 'JavelinThrow': 45, 'CricketShot': 24, 'FrisbeeCatch': 31, 'HammerThrow': 36, 'BaseballPitch': 7, 'TennisSwing': 92, 'CliffDiving': 22, 'CricketBowling': 23, 'Diving': 26, 'LongJump': 51, 'GolfSwing': 33, 'Shotput': 79, 'SoccerPenalty': 85}

video_id_mapping = {}
video_iter_list = [0,0,0]
video_duration = collections.defaultdict(lambda: 1.0)

def time_to_frame_num(i,t):
	return math.floor(float(t)/(video_duration[i]/50))

dir_list = ["./../annotation/", "./../test_annotation/"]
for dir_name in dir_list:
	for filename in os.listdir(dir_name):
		if("Ambiguous" not in filename):
			f = open(os.path.join(dir_name, filename))
			for line in f:
				video_name, _, start_time, end_time = line.replace("\n","").split(" ")
				if(video_name in train_list):
					idx = 0
				elif(video_name in val_list):
					idx = 1
				else:
					idx = 2
				if(video_name not in video_id_mapping):
					video_id_mapping[video_name] = video_iter_list[idx]
					video_iter_list[idx] += 1
					meta[idx][video_id_mapping[video_name]] = {}
					meta[idx][video_id_mapping[video_name]]['vidName'] = video_name
					#meta[idx][video_iter]['seq'] = [start_frame, end_frame]
					meta[idx][video_id_mapping[video_name]]['dets'] = {}
					for action in action_id_mapping:
						meta[idx][video_id_mapping[video_name]]['dets'][action_id_mapping[action]] = {}
				if(len(meta[idx][video_id_mapping[video_name]]['dets'][action_id_mapping[filename.split("_")[0]]]) == 0):
					meta[idx][video_id_mapping[video_name]]['dets'][action_id_mapping[filename.split("_")[0]]] = [[time_to_frame_num(video_name, start_time), time_to_frame_num(video_name, end_time)]]
				else:
					meta[idx][video_id_mapping[video_name]]['dets'][action_id_mapping[filename.split("_")[0]]].append([time_to_frame_num(video_name, start_time), time_to_frame_num(video_name, end_time)])

with open('train_meta_file.json', 'w') as outfile:
    json.dump(meta[0], outfile)

with open('val_meta_file.json', 'w') as outfile:
    json.dump(meta[1], outfile)

with open('test_meta_file.json', 'w') as outfile:
    json.dump(meta[2], outfile)