from ipdb import set_trace
import json
import os
import pandas as pd
import numpy as np
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image as Image
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained Faster-RCNN')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="./PennFudanPed/PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="./PennFudanPed/PennFudanPed_prediction.json")
    args = parser.parse_args()
    return args

def get_device(gpu_no=0):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_no)
		return torch.device('cuda:{}'.format(gpu_no))
	else:
		return torch.device('cpu')

class PennFudan_dataset(Dataset):
    def __init__(self, root, test_json):
        super(PennFudan_dataset, self).__init__()
        self.root = root
        
        test_json = json.loads(open(test_json,'r').read())
        self.img_dicts = test_json['images']
        
        self.normalize_trnsfrm = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_dict = self.img_dicts[index]
        img = Image.open(os.path.join(self.root, img_dict["file_name"]))
        x = self.normalize_trnsfrm(img)
        return x, img_dict["id"]
  
    def __len__(self):
        return len(self.img_dicts)

def main(root, test_json, output_json, device):
    # pretrained faster rcnn model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
    model = model.to(device)
    model.eval()

    # dataset to evaluate
    testset = PennFudan_dataset(root, test_json)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # predictions will be saved iteratively
    predictions = []
    no_pred_count = 0
    nms_count = 0

    # for saving images with predicted bboxes, and comparing them with annotations
    annotations = test_json['annotations'] # this is ONLY used for comparison of predicted bboxes
    annotations = pd.json_normalize(annotations)
    save_preds_dir = os.path.join(args.root, "predictions_faster_rcnn")
    if os.path.exists(save_preds_dir) == False:
        os.mkdir(save_preds_dir)
    transfrm = transforms.ToPILImage()

    print("\nstarting inference over given test.json")
    for batch_idx, (imgs,img_ids) in enumerate(testloader):
        imgs = imgs.to(device)
        outputs = model(imgs)

        # for each prediction (for image) iterate over possible bb and append them
        # for batch size = 1, there will be only one output
        for idx, output in enumerate(outputs):
            img = imgs[idx].cpu()
            img = np.array(transfrm(img))

            img_id = img_ids[idx].item()

            labels = output['labels'].cpu()
            if len(labels == 1) == 0:
                no_pred_count += 1
                continue

            bboxes = output['boxes'].cpu()[labels == 1].astype(int) # .tolist()
            scores = output['scores'].cpu()[labels == 1].astype(float) # .tolist()
            
            if len(scores) != 0:
                # do NMS and append the predictions in COCO format
                init = len(scores)
                bboxes, scores = do_NMS(bboxes, scores, overlapThresh=0.65) # bboxes.dtype is int, scores.dtype is float
                final = len(scores)
                nms_count += (init-final)

            if len(scores) == 0:
                # no predictions
                # print("no prediction encountered")
                no_pred_count+=1
                continue
            
            # set_trace()
            for bb, score in zip(bboxes, scores):
                pred = {}
                pred["image_id"] = img_id
                pred["score"] = float(score)
                pred["category_id"] = 1
                pred["bbox"] = bb.astype(float).tolist()
                predictions.append(pred)
            
            # for visualization of bboxes and comparison with annotations
            save_img_with_pred(img, img_id, bboxes, scores, list(annotations.loc[annotations['image_id'] == img_id]['bbox']), save_preds_dir)
    
    print("no predictions for %u images out of %u"%(no_pred_count, len(testset)))
    with open(output_json, "w") as f:
        json.dump(predictions, f)

    print("Non-Maximal Suppression reduced %u Bounding Boxes"%(nms_count))

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    main(args.root, args.test, args.out, device)