from mmdet.apis import init_detector, inference_detector

# config_file = 'faster_rcnn_r50_fpn_1x_coco.py'
config_file = 'configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = '/home/ubuntu/kartik/CBNetV2/checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
output = inference_detector(model, 'demo/demo.jpg')
model.show_result('demo/demo.jpg', output, out_file='demo_res.jpg')
print(output)