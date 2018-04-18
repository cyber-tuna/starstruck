import cv2
import os
import numpy as np
import argparse
import time
import sys
import progressbar

parser = argparse.ArgumentParser(description="Star Trail Photo Stacker")

parser.add_argument('-c', '-comet', type=int, action='store', dest='comet_length', metavar='<Comet Length>', 
					help='Length of the comet trail', default='-1')

parser.add_argument('-i', '-input', type=str, action='store', dest='input_dir', metavar='<Path Name>', 
					help='Path to directory containing the photos', default='none', required=True)

parser.add_argument('-ov', type=str, action='store', dest='output_vid', metavar='<Video Name>', 
					help='Name to be given to video file output (must have .avi extension)', default='none')

arg_results = parser.parse_args()

image_folder = arg_results.input_dir

images = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(arg_results.output_vid, cv2.VideoWriter.fourcc('M','J','P','G'), 13, (width,height))

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', (int(width/3), int(height/3)))
cv2.moveWindow("result", 1000,1000);

images.sort()
images.reverse()

result_img = None
old_img = None
counter = 0
diff = None

# bar = progressbar.progressbar(range(len(images)))

print("Processing", len(images), "images...")
with progressbar.ProgressBar(max_value=len(images)) as bar:

	for image in images:
		counter += 1
		
		if result_img is None:
			result_img = cv2.imread(os.path.join(image_folder, image))
			old_img = result_img
		else:
			new = cv2.imread(os.path.join(image_folder, image))
			# new = cv2.GaussianBlur(new, (3,3), 0, 0 )
			
			# result_img[:,:,1] = max(new[:,:,1], result_img[:,:,1])
			# result_img[:,:,2] = max(new[:,:,2], result_img[:,:,2])

			# diff = cv2.absdiff(new, old_img)

			# kernel_erode = np.ones((8,8),np.uint8)
			# kernel_dilate = np.ones((10,10),np.uint8)
	           
			# diff = cv2.dilate(diff,kernel_dilate,iterations = 3)
			# diff = cv2.GaussianBlur(diff, (25,25), 0, 0 )
			# # retval, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
			# diff = cv2.erode(diff,kernel_erode,iterations = 3)

			if arg_results.comet_length != -1:
				hsv = cv2.cvtColor(result_img, cv2.COLOR_RGB2HSV)
				hsv[:,:,2] = np.maximum(hsv[:,:,2] - arg_results.comet_length, 0)
				result_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

			result_img[:,:,0] = np.maximum(new[:,:,0], result_img[:,:,0])
			result_img[:,:,1] = np.maximum(new[:,:,1], result_img[:,:,1])
			result_img[:,:,2] = np.maximum(new[:,:,2], result_img[:,:,2])

			if arg_results.output_vid != 'none':
				video.write(result_img)

			cv2.imshow('result', result_img)
			cv2.waitKey(1)

			old_img = new
			bar.update(counter)

cv2.imwrite('result_img.jpg',result_img)

	# print("writing")
	# video.write(img)
	# video.write(img)
	# video.write(img)
	# video.write(img)
	# video.write(img)
	# video.write(img)

cv2.destroyAllWindows()
video.release()