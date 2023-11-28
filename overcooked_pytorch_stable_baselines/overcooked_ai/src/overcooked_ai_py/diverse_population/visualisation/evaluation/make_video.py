import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path_to_images="../../../../../../../five_checkpoints/"
path_to_videos="../../../../"
image_files = []

for img_number in range(0,1999):
    image_files.append(path_to_images + str(img_number) + '.png')

fps = 3

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path_to_videos + 'five_checkpoints.mp4')