from mmdet.apis import init_detector, inference_detector
import cv2
import time
import os

if __name__ == '__main__':
	config_file = 'configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
	checkpoint_file = 'checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
	
	model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
	images_dir = '/home/ubuntu/kartik/STEGO/datasets/cocostuff/images/test2017'
	result_dir = '/home/ubuntu/kartik/STEGO/results/predictions/quicktest/cbnet'
 
	os.makedirs(result_dir, exist_ok=True)
	
	start_time = time.time()
 
	for f_path in sorted(os.listdir(images_dir))[:100]:
		print(f_path)
		res_path = os.path.join(result_dir, f_path)
		im_path = os.path.join(images_dir, f_path)
		output = inference_detector(model, im_path)
		result, segms = model.show_result(im_path, output, score_thr=0.3, only_segmentation=False)
		cv2.imwrite(res_path, result)
	
	end_time = time.time()
 
	print(end_time - start_time)