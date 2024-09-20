import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from PIL import Image

from robust_segment_anything import sam_model_registry
from robust_segment_anything import sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide 

def show_boxes(coords, ax):
    x1, y1, x2, y2 = coords
    width = x2-x1
    height = y2-y1    
    bbox = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)
   
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
        
    h, w = mask.shape[-2:]    
    mask = mask.detach().cpu().numpy()   
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
opt = argparse.Namespace()
opt.case = "clear"
opt.gpu = 0
opt.model_size = 'l'
opt.checkpoint_path = "robustsam_checkpoint_l.pth"
opt.checkpoint_path = 'robustsam_checkpoint_{}.pth'.format(opt.model_size)    

def create_sam_model():
    sam_model = sam_model_registry["vit_{}".format(opt.model_size)](opt=opt, checkpoint=opt.checkpoint_path)
    sam_model = sam_model.to(opt.gpu)
    print('Succesfully loading model from {}'.format(opt.checkpoint_path))
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    return sam_model, sam_transform

print('Use bounding box as prompt!')

def robust_sam(image_path, box_prompt, sam_model, sam_transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(opt.gpu)
    image_t = torch.permute(image_t, (0, 3, 1, 2))
    image_t_transformed = sam_transform.apply_image_torch(image_t.float())

    data_dict = {}      
    box_t = torch.Tensor(box_prompt).unsqueeze(0).to(opt.gpu)
    data_dict['image'] = image_t_transformed
    data_dict['boxes'] = sam_transform.apply_boxes_torch(box_t, image_t.shape[-2:]).unsqueeze(0)          
    data_dict['original_size'] = image_t.shape[-2:]  

    with torch.no_grad():   
        batched_output = model.predict(opt, [data_dict], multimask_output=False, return_logits=False)    

    output_mask = batched_output[0]['masks']
    plt.figure(figsize=(10,10))
    plt.imshow(image[:, :, ::-1])

    show_boxes(box_prompt, plt.gca())
    show_mask(output_mask[0][0], plt.gca())
    plt.axis('off')
    plt.savefig("after_robust.png", bbox_inches='tight')

    print('Finish inferencing...')

    output_mask = output_mask.squeeze(0).squeeze(0)
    # Chuyển tensor sang numpy array
    numpy_image = output_mask.cpu().numpy()
    # Chuyển đổi giá trị về khoảng [0, 255] nếu cần (tùy thuộc vào phạm vi của tensor ban đầu)
    numpy_image = (numpy_image * 255).astype(np.uint8)
    mask_pil_image = Image.fromarray(numpy_image)
    mask_pil_image.save("mask_pil_image.png")
    return mask_pil_image