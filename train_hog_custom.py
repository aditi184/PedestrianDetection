import json
import os
import cv2
from ipdb import set_trace
import random
import numpy as np
import pandas as pd
import argparse
from utils import *
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import pickle
from sklearn import model_selection

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--train', type=str, default="PennFudanPed_train.json")
    parser.add_argument('--save_model', type=str, default="hog_custom.pt")
    args = parser.parse_args()
    return args

def get_patch(image, bb):
    x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # w is along x (right), h is along y (down)
    return image[y:y+h+1, x:x+w+1, :]

def create_positive_samples(root, train_json):
    # extract patches containing pedestrians using given annotations, resize them, and save inside "root/PennFudanPed/Positive"
    positive_samples = []
    positive_dir = os.path.join(os.path.join(root, "PennFudanPed"), "Positive")
    if os.path.exists(positive_dir) == False:
        os.mkdir(positive_dir)

    img_dicts = train_json['images']
    annotations = train_json['annotations']
    annotations = pd.json_normalize(annotations)

    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        bboxes = list(annotations.loc[annotations['image_id'] == img_id]['bbox'])
        positive_samples += extract_patches(img, img_id, bboxes, positive_dir)
    
    return positive_samples

def extract_patches(img, img_id, bboxes, positive_dir, patch_size=(64,128)):
    pos_samples = []
    for idx, bbox in enumerate(bboxes):
        patch = get_patch(img, bbox)
        patch = cv2.resize(patch, patch_size)
        save_patch = os.path.join(positive_dir, str(img_id)+"_%u.jpg"%(idx))
        
        pos_samples.append(patch)
        cv2.imwrite(save_patch, patch)
        
    return pos_samples

def create_negative_samples(root,train_json):
    negative_dir = os.path.join(os.path.join(root, "PennFudanPed"), "Negative")
    negative_samples = []
    if os.path.exists(negative_dir) == False:
        os.mkdir(negative_dir)
    
    img_dicts = train_json['images']
    annotations = train_json['annotations']
    annotations = pd.json_normalize(annotations)
    
    for img_dict in img_dicts:
        img = cv2.imread(os.path.join(root,img_dict['file_name']))
        img_id = img_dict['id']
        bboxes = list(annotations.loc[annotations['image_id'] == img_id]['bbox'])
        negative_samples += extract_neg_patches(img, img_id, bboxes, negative_dir)
    
    return negative_samples

def extract_neg_patches(img, img_id, bboxes, negative_dir, patch_size=(64,128), max_samples=5):
    neg_samples = []
    x_list = np.random.randint(0, img.shape[1]-patch_size[0], max_samples)
    y_list = np.random.randint(0, img.shape[0]-patch_size[1], max_samples)
    assert len(x_list) == len(y_list) == max_samples, set_trace

    # select only those sampled patches that don't overlap with bboxes
    for idx,(x, y) in enumerate(zip(x_list, y_list)):
        if iou_less(x, y, patch_size, bboxes, overlap_max=0.1):
            patch = img[y:y+patch_size[1],x:x+patch_size[0],:]
            neg_samples.append(patch)
            save_patch = os.path.join(negative_dir, str(img_id)+"_%u.jpg"%(idx))
            cv2.imwrite(save_patch, patch)

    return neg_samples

def iou_less(x, y, patch_size, bboxes, overlap_max=0.1):
    # check whether the patch overlaps with all bboxes or not
    pred_box = [x,y,x+patch_size[0],y+patch_size[1]]
    for bb in bboxes:
        x1, y1, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        true_box = [x1,y1,x1+w,y1+h]

        px1 = max(true_box[0], pred_box[0])
        py1 = max(true_box[1], pred_box[1])
        px2 = min(true_box[2], pred_box[2])
        py2 = min(true_box[3], pred_box[3])
        intersection = max(0, px2 - px1 + 1) * max(0, py2 - py1 + 1)
        area1 = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
        area2 = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
        union = area1 + area2 - intersection
        iou = intersection / float(union)

        if iou >= overlap_max:
            return False
    return True

def main(root, train_json, save_model):
    train_json = json.loads(open(train_json,'r').read())

    # create training data ie positive and negative samples for SVM using train_json
    positive_samples = create_positive_samples(root, train_json)
    negative_samples = create_negative_samples(root, train_json)[:len(positive_samples)]
    pos_labels = [1] * len(positive_samples)
    neg_labels = [-1] * len(negative_samples)

    print("Training SVM...\n")
    print("number of positive samples:%u , number of negative samples:%u"%(len(positive_samples), len(negative_samples)))

    samples = np.concatenate((positive_samples,negative_samples), axis=0)
    labels = np.hstack((pos_labels,neg_labels))

    x_train, x_test, y_train, y_test = model_selection.train_test_split(samples,labels,test_size = 0.3, random_state=4)
    hog_features = []
    for x in x_train:
        x_feature = hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=True)
        hog_features.append(x_feature)

    hog_features = np.array(hog_features)
    clf = svm.SVC()
    clf.fit(hog_features,y_train)

    test_features = []
    for i in x_test:
        feature = hog(i,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=True)
        test_features.append(feature)

    y_pred = clf.predict(test_features)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))
    model_path = os.path.join(root, args.save_model)
    print("model saved at:" , model_path)
    pickle.dump(clf, open(model_path, 'wb'))

if __name__ == "__main__":
    fix_seed(seed=4)
    args = parse_args()
    main(args.root, args.train, args.save_model)