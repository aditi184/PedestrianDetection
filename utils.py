import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def draw_rectangles(img, bboxes, scores):
    for idx, (x, y, w, h) in enumerate(bboxes):
        cv2.rectange(img, (x,y), (x+w,y+h), (0,255,0), 2)
    return img

def do_NMS(bboxes, scores, overlapThresh):
    # changes x,y,w,h to x,y,x2,y2
    for idx in range(bboxes.shape[0]):
        bboxes[idx, 2] += bboxes[idx, 0]
        bboxes[idx, 3] += bboxes[idx, 1]
    
    bboxes_nms = non_max_suppression(bboxes, probs=None, overlapThresh=overlapThresh)
    
    # get scores for these bounding boxes
    scores_nms = []
    for bb in bboxes_nms:
        scores_nms.append(scores[(bb == bboxes).mean(axis=1) == 1][0,0])
    scores_nms = np.array(scores_nms).reshape(-1,1)

    # changes x,y,x2,y2 to x,y,w,h
    for idx in range(bboxes_nms.shape[0]):
        bboxes_nms[idx, 2] = bboxes_nms[idx, 2] - bboxes_nms[idx, 0] + 1
        bboxes_nms[idx, 3] = bboxes_nms[idx, 3] - bboxes_nms[idx, 1] + 1

    return bboxes_nms, scores_nms

def save_img_with_pred(img, img_id, bboxes, scores, annotations, save_preds_dir):
    for idx, (x,y,w,h) in enumerate(bboxes):
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.putText(img, str(scores[idx]), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0))
    
    for idx, (x,y,w,h) in enumerate(annotations):
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
    cv2.imwrite(os.path.join(save_preds_dir, str(img_id)+".jpg"), img)