import shutil
import argparse
import os
from landmarks_extractor import LandmarkDetector

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("imgpath", help="Path to the input image")
parser.add_argument("--bg_color", help="Provide the hexcode string for background.")

args = parser.parse_args()

bgc = "#000000"

if args.bg_color:    
    bgc = args.bg_color

print("BG Color: ", args.bg_color)
print("Path: ", args.imgpath)


filename = str(os.path.basename(args.imgpath))[:-4]

os.mkdir(filename + "_src")

Landmarker = LandmarkDetector(args.imgpath, src_dir=f"{filename}_src", out_dir=f"{filename}_src", inputImage=f"{filename}")

Landmarker.detect()
Landmarker.find_closed_mouth_landmark()
Landmarker.use_inner_lips_only()


