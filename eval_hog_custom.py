from ipdb import set_trace
import json
import os
import cv2
import numpy as np
import argparse
from ipdb import set_trace
import pandas as pd
from tqdm import tqdm
from utils import *
import pickle
from skimage.feature import hog
import imutils

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="PennFudanPed_predict.json")
    parser.add_argument('--model',type=str, default="hog_custom.pt")
    parser.add_argument('--num_pyr_lyrs',type=int, default=1, help='number of pyramid layers')
    args = parser.parse_args()
    return args

def make_predictions(clf, root, test_json, output_json, num_pyr_lyrs, patch_size=(64, 128)):
    # predictions will be saved iteratively
    predictions = []
    no_pred_count = 0
    nms_count = 0
    sigmoid = torch.nn.Sigmoid() # use sigmoid to normalize svm scores

    # for saving images with predicted bboxes, and comparing them with annotations
    annotations = test_json['annotations'] 
    annotations = pd.json_normalize(annotations)
    save_preds_dir = os.path.join(args.root, "predictions_hog_custom")
    if os.path.exists(save_preds_dir) == False:
        os.mkdir(save_preds_dir)

    # read the images using the file name in the json file
    print("\nstarting inference over given test.json")
    img_dicts = test_json['images']

    # detect multiscale hyperparameters
    winStride = (12, 24)
    padding = (10, 10)

    for img_dict in tqdm(img_dicts):
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        
        # go into various pyramid levels, get all the predicted bbs
        # after getting all possible bbs, apply nms finally
        # img_list = [img]
        bboxes, scores = [], []
        curr_img = img
        for level_num in range(num_pyr_lyrs):
            if level_num != 0:
                curr_img = cv2.pyrDown(curr_img)

            for y in range(0, curr_img.shape[0] - patch_size[1], winStride[1]):
                for x in range(0, curr_img.shape[1] - patch_size[0], winStride[0]):
                    patch = curr_img[y:y+patch_size[1], x:x+patch_size[0]]
                    if patch.shape[0] < patch_size[1] or patch.shape[1] < patch_size[0]:
                        continue
                    
                    hog_descriptor = hog(
                        patch, 
                        orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(3, 3), block_norm='L2-Hys', 
                        visualize=False, transform_sqrt=False, 
                        feature_vector=True, multichannel=True
                    )

                    hog_descriptor = hog_descriptor.reshape(1, -1)
                    svm_pred = clf.predict(hog_descriptor)
                    if svm_pred[0] == 1:
                        svm_score = abs(clf.decision_function(hog_descriptor)[0])
                        # set_trace()
                        x1 = x * (2 ** level_num)
                        y1 = y * (2 ** level_num)
                        w = patch_size[0] * (2 ** level_num)
                        h = patch_size[1] * (2 ** level_num)
                        bbox = [x1, y1, w, h]
                        bboxes.append(bbox)
                        scores.append(svm_score)
            
        bboxes = np.array(bboxes).astype(int)
        scores = np.array(scores).astype(float).reshape(-1)

        if len(scores) != 0:
            # do NMS and append the predictions in COCO format
            init = len(scores)
            bboxes, scores = do_NMS(bboxes, scores, overlapThresh=0.8) # bboxes.dtype is int, scores.dtype is float
            final = len(scores)
            nms_count += (init-final)
        
        if len(scores) == 0:
            # no predictions
            # print("no prediction encountered")
            no_pred_count+=1
            continue

        for bb, score in zip(bboxes, scores):
            pred = {}
            pred["image_id"] = img_id
            pred["score"] = sigmoid(torch.tensor(float(score))).item()
            pred["category_id"] = 1
            pred["bbox"] = bb.astype(float).tolist()
            predictions.append(pred)
        
        # for visualization of bboxes and comparison with annotations
        save_img_with_pred(img, img_id, bboxes, scores, list(annotations.loc[annotations['image_id'] == img_id]['bbox']), save_preds_dir)
    
    print("no predictions for %u images out of %u"%(no_pred_count, len(img_dicts)))
    with open(output_json, "w") as f:
        json.dump(predictions, f)

    print("Non-Maximal Suppression reduced %u Bounding Boxes"%(nms_count))

def main(root, test_json, output_json, num_pyr_lyrs):
    # pretrained hog model
    clf = pickle.load(open(args.model, 'rb'))
    make_predictions(clf,root,test_json, output_json, num_pyr_lyrs, patch_size=(64,128))# (120, 240)

if __name__ == "__main__":
    fix_seed(seed=4)
    args = parse_args()
    test_json = json.loads(open(args.test,'r').read())
    main(args.root, test_json, args.out, args.num_pyr_lyrs)