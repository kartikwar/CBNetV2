from mmdet.apis import init_detector, inference_detector
import cv2
import time
import numpy as np
import imutils
import itertools
import os

def sad_calculation(mask, lookup):
	# mask = cv2.resize(mask.astype('float64'), (500,500))
	# lookup = cv2.resize(lookup, (500, 500))

	# assert(mask.shape == lookup.shape)
	# assert(mask.dtype == lookup.dtype)

	mse_diff = ((mask - lookup) ** 2).sum()
	sad_diff = np.abs(mask - lookup).sum()

	# print(sad_diff)

	return sad_diff, mse_diff

# to check if r2 is inside r1
def contains(r1, r2):
	return r1[0] < r2[0] < r2[2] < r1[2] and r1[1] < r2[1] < r2[3] < r1[3]

def remove_contained_cnts(cnts):
	cnt_indexes = list(itertools.combinations(range(len(cnts)), 2))

	remove_indexes = []

	keep_indexes = []

	cnts_ = []

	for index_pair in cnt_indexes:
		indexes = [index_pair[0], index_pair[1]]
		cnt1, cnt2 = cnts[index_pair[0]], cnts[index_pair[1]]
		rect_0 = cv2.boundingRect(cnt1)
		rect_0 = (rect_0[0], rect_0[1], rect_0[0] + rect_0[2], rect_0[1] + rect_0[3])
		rect_1 = cv2.boundingRect(cnt2)
		rect_1 = (rect_1[0], rect_1[1], rect_1[0] + rect_1[2], rect_1[1] + rect_1[3])

		# rect1 inside rect0
		if contains(rect_0, rect_1):
			#             print(rect_0, rect_1)
			remove_indexes.append(index_pair[1])

		# rect0 inside rect1
		elif contains(rect_1, rect_0):
			#             print(rect_1, rect_0)
			remove_indexes.append(index_pair[0])

	keep_indexes = [ind for ind in range(len(cnts)) if ind not in remove_indexes]

	cnts_ = [cnts[ind] for ind in keep_indexes]

	return cnts_

def get_contours(mask):
	masks = []
	mask = mask * 255
	mask = mask.astype('uint8')
	# cv2.imwrite('current_mask.jpg', mask)
	
	"""
	draw a black line on all 4 corners of the image,
	this is done to avoid open contours later on
	"""
	height, width = mask.shape
	cv2.line(mask, (0, 0), (0, height), (0, 0, 0), thickness=5)
	cv2.line(mask, (0, 0), (width, 0), (0, 0, 0), thickness=5)
	cv2.line(mask, (0, height), (width, height), (0, 0, 0), thickness=5)
	cv2.line(mask, (width, 0), (width, height), (0, 0, 0), thickness=5)
	
	edged = cv2.Canny(mask, 10, 30)
	kernel = np.ones((5, 5), np.uint8)
	edged = cv2.dilate(edged, kernel, iterations=1)

	cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	
	cnts = remove_contained_cnts(cnts)
	
	for cnt in cnts:
		img_cnt = np.zeros((mask.shape[0], mask.shape[1]))  # create a single channel 200x200 pixel black image plt.imshow(img_cnt)
		img_cnt = cv2.fillPoly(img_cnt, pts=[cnt], color=(255, 255, 255))
		# masks.append((np.logical_and(img_cnt, mask)*255).astype('uint8'))
		masks.append(np.logical_and(img_cnt, mask))
		# cv2.imwrite('contour.jpg', np.logical_and(img_cnt, mask)*255)

	return masks

def remove_noise(mask, optimized_height, saliency_mask):
	result = []
	encountered_pixels = []
	
	mask = cv2.resize(mask.astype('uint8'), (optimized_height, optimized_height))
	mask = mask.astype('bool')
	cnts = mask
	#to do: add cnt logic if reqd
	# cnt = mask
	
	#function to split mask into separate contours
	start_time = time.time()
	cnts = get_contours(mask)
	end_time = time.time()
	# print(end_time - start_time)

	for cnt in cnts:
		lookup = np.where(cnt==True, saliency_mask/255.0 , 0)
		total_pixels = np.count_nonzero(cnt)
		
		# lookup = np.where(lookup > 0.05, lookup, 0)
		
		if not np.all(lookup == 0.) and total_pixels > 0:
			start_time = time.time()
			sad_diff, mse_diff = sad_calculation(cnt, lookup)
			end_time = time.time()
			# print(end_time - start_time)
			# cv2.imwrite('lookup.jpg', lookup*255)
			# cv2.imwrite('mask.jpg', cnt*255)
			# lookup = saliency_mask[mask]
			# lookup_pixels = np.count_nonzero(lookup)
			thresh = sad_diff/total_pixels
			confidence = 1. - thresh
			encountered_pixels.append(cnt)
			
			#hyperparameter (to do : experiment with diff values)
			#original is 0.85, stable is 0.7
			# if confidence > 0.7:
			# if confidence > 0.85:
			if confidence > 0.8:
			# if 0.7 < confidence < 0.85:
				# result[mask] = mask
				result.append(cnt)     
				# append_count += 1
	
	return result, encountered_pixels

def optimize_saliency(im_path, saliency_path):
	OPTIMIZE_HEIGHT = 500
	start_time = time.time()
	output = inference_detector(model, im_path)
	img, segms = model.show_result(im_path, output, score_thr=0.)
	# print(segms.shape)
	# img, segms = model.show_result(im_path, output, score_thr=0.1)
	# return img
	
	result = []
	encountered_pixels = []
	
	saliency_mask = cv2.imread(saliency_path, 0)
	
	
	
	# return img
	# cv2.imwrite('image.png', img)
	saliency_mask = cv2.resize(saliency_mask, (OPTIMIZE_HEIGHT, OPTIMIZE_HEIGHT))
	# result = np.zeros(img.shape[:2])

	#to do: need to fix this
	if segms is not None:
		# to do: check if this could be converted into multiprocessing 
  		# or can you make use of numpy here
		for mask in segms:
			curr_result, seen_pixels = remove_noise(mask, OPTIMIZE_HEIGHT, saliency_mask)
			result = result + curr_result
			encountered_pixels = encountered_pixels + seen_pixels
	
 
	#combine encountered pixels and saliency mask
	if len(result) > 0:
		result = np.array(result)
		result = np.sum(result, axis=0)
		result = np.where(result>0, 1, 0)
		result = result*255
		result = result.astype('uint8')
	else:
		result = np.zeros(img.shape[:2])

	if len(encountered_pixels) > 0:
		encountered_pixels = np.array(encountered_pixels)
		encountered_pixels = np.sum(encountered_pixels, axis=0)
		encountered_pixels = np.where(encountered_pixels>0, 1, 0)
		encountered_pixels = encountered_pixels*255
		encountered_pixels = encountered_pixels.astype('uint8')
	else:
		encountered_pixels = np.zeros(img.shape[:2])

	corrected_result = np.where(encountered_pixels > 0, result , saliency_mask)
 
	#multiply with saliency mask
	saliency_mask = saliency_mask.astype('float32')/255.0
	corrected_result = corrected_result.astype('float32')/255.0
	corrected_result = saliency_mask * corrected_result
	corrected_result = corrected_result*255
	
	return cv2.resize(corrected_result.astype('uint8'), (img.shape[1], img.shape[0]))

if __name__ == '__main__':
	config_file = 'configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
	checkpoint_file = 'checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
	
	model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
	images_dir = '../datasets/test-dataset/images/'
	# images_dir = '/home/ubuntu/kartik/Detic/results/inputs/imgs'
	saliency_dir = '../bg_remove_saliency_training/results/v15_asp_resized/'
	# saliency_dir = '/home/ubuntu/kartik/Detic/results/inputs/saliency_masks'
	result_dir = 'results/cbnet-raw'
 
	os.makedirs(result_dir, exist_ok=True)
	
	start_time = time.time()
 
	for f_path in sorted(os.listdir(images_dir))[:100]:
		print(f_path)
		res_path = os.path.join(result_dir, f_path)
		if not os.path.exists(res_path):
		# if f_path == '139412.jpg':
			im_path = os.path.join(images_dir, f_path)
			sal_path = os.path.join(saliency_dir, f_path.replace('.jpg', '_sal_fuse.png'))
			# sal_path = os.path.join(saliency_dir, f_path.replace('.jpg', '.png'))
			result = optimize_saliency(im_path, sal_path)
			cv2.imwrite(res_path, result)
	
	end_time = time.time()
 
	print(end_time - start_time)