# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import numpy as np
from os.path import join as opj
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def evaluation(logger, cfg, model, test_loader, affordance):
    num_classes = len(affordance)
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(len(affordance))]
    total_correct_class = [0 for _ in range(len(affordance))]
    total_iou_deno_class = [0 for _ in range(len(affordance))]
    with torch.no_grad():
        model.eval()
        for i,  temp_data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            (data, _, label, _, _) = temp_data

            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)        # B x 3 x N
            label = torch.squeeze(label).cpu().numpy()    # B x N
            B = label.shape[0]
            N = label.shape[1]
            
            afford_pred,_,_ = model(data, affordance)
            afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()    # B x N x 18
            afford_pred = np.argmax(afford_pred, axis=2)    # B x N
            correct = np.sum((afford_pred == label))
            total_correct += correct
            total_seen += (B * N)
            for i in range(num_classes):
                total_seen_class[i] += np.sum((label == i))
                total_correct_class[i] += np.sum((afford_pred == i) & (label == i))
                total_iou_deno_class[i] += np.sum((afford_pred == i) | (label == i))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        logger.cprint('eval point avg class IoU: %f' % (mIoU))
        logger.cprint('eval point accuracy: %f' % (total_correct / float(total_seen)))
        logger.cprint('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(num_classes):
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    affordance[l], total_correct_class[l] / float(total_iou_deno_class[l]))
        logger.cprint(iou_per_class_str)
    return mIoU