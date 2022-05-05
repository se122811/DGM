import os
import sys
import math
import numpy as np
import cv2
import torch
import data
from options.test_options import TestOptions
import models
import matplotlib.pyplot as plt
import copy

def draw4images(images, img_path):
    w = 5; h=5; rows=1; cols=2; axes=[]; fig=plt.figure()
    for i in range(rows*cols):
        axes.append(fig.add_subplot(rows,cols,i+1))
        plt.imshow(images[i])

    axes[0].set_title = ("Original Image")
    axes[1].set_title = ("Generated Image")
    fig.tight_layout()
    plt.savefig(img_path)



opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

# test
num = 0



for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    with torch.no_grad():
        generated,_ = model(data_i, mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        pred_im = generated[b].transpose((1,2,0))
        # print('process image... %s' % img_path[b])
        def visualize():
            bbox = data_i['mask'].reshape(256,256,1).detach().cpu().numpy()
            bbox = np.repeat(bbox, 3, axis=2) * 50
            img = data_i['image'].detach().cpu().numpy().transpose(2,3,0,1).reshape(256,256,3)
            img = np.clip((img+1)/2*255 + bbox, 0, 255).astype(np.uint8)
            return bbox, img

            
        img, pred_im, img2, pred_im2 = error(pred_im)
        # bbox, img = error(pred_im)
        
        # draw4images([img,pred_im], img_path[b])

    
    # pred_im = pred_im + bbox
    # cv2.imwrite(img_path[b], pred_im[:,:,::-1])
