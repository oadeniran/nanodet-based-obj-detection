import os
import cv2
import torch
from nanodet.util import cfg, load_config, Logger
from demo.demo import Predictor
from nanodet.util import overlay_bbox_cv
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cpu')

config_path = 'nanodet/config/nanodet-plus-m_416.yml'
model_path = 'nanodet/nanodet-plus-m_416_checkpoint.ckpt'
load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, model_path, logger, device=device)

def make_inference_img(image_path):
    meta, res = predictor.inference(image_path)
    result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.5)
    cv2.imwrite("static/inf-img.jpg", result)
    return "Done"

def make_inference_frame(frame):
    temp = "frame.jpg"
    cv2.imwrite(temp,frame)
    meta, res = predictor.inference(temp)
    result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.5)
    resp = {
            'res': result.tolist(),
        }
    return resp
    