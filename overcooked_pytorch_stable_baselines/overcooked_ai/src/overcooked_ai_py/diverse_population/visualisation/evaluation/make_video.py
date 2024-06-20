import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_directory", default=None, type=str,help="path to directory with images for the video" )
parser.add_argument("--video_path", default=None, type=str, help="where to save the video")
args = parser.parse_args([] if "__file__" not in globals() else None)


fps = 3



if __name__ == "__main__":
    dir_files = [f for f in listdir(args.img_directory) if isfile(join(args.img_directory, f)) and f.endswith('.png')]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(dir_files, fps=fps)
    clip.write_videofile(args.video_path + '.mp4')