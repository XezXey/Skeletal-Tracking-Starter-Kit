import numpy as np
import cv2
import os
import sys
import subprocess
import glob, tqdm

out_path = './data/images/'
os.makedirs(out_path, exist_ok=True)

def proc(path):
    for vid in tqdm.tqdm(glob.glob(os.path.join(path, "*.mp4"))):  # FIXED: glob.glob requires a full pattern
        # print(vid)
        name = vid.split("/")[-1].split(".")[0]
        op = os.path.join(out_path, name)
        os.makedirs(op, exist_ok=True)
        cmd = f'ffmpeg -i "{vid}" "{op}/%05d.jpg"'  # quote paths to handle spaces
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




if __name__ == "__main__":
    proc(r"/data/mint/FIFA_Skeletal_Dataset/FIFA Skeletal Light - Camera 1 Footage/ChallengeTest/compressed/")
    proc(r"/data/mint/FIFA_Skeletal_Dataset/FIFA Skeletal Light - Camera 1 Footage/ChallengeVal/compressed/")
