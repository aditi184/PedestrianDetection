from ipdb import set_trace
import json
import os
import cv2
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
    for batch_idx, (imgs,img_ids) in enumerate(testloader):
        imgs = imgs.to(device)
        outputs = model(imgs)

        # for each prediction (for image) iterate over possible bb and append them
        for idx, output in enumerate(outputs):
            img_id = img_ids[idx].item()

            labels = output['labels'].cpu()
            if len(labels) != 0:
                bboxes = output['boxes'].cpu()[labels == 1]
                scores = output['scores'].cpu()[labels == 1]

                if len(scores) == 0:
                    bboxes, scores = np.array([[0,0,0,0]]), np.array([[0]])
                    print("no person prediction encountered")
            else:
                bboxes, scores = np.array([[0,0,0,0]]), np.array([[0]])
                print("no prediction encountered")
            
            # set_trace()
            for bb, score in zip(bboxes, scores):
                pred = {}
                pred["image_id"] = img_id
                pred["score"] = score.item()
                pred["category_id"] = 1
                pred["bbox"] = bb.tolist()
                predictions.append(pred)
    
    with open(output_json, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    main(args.root, args.test, args.out, device)