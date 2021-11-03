import pandas as pd
import json
import os
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection using pretrained HoG Person Detector')
    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', type=str, default="./PennFudanPed/PennFudanPed_val.json")
    parser.add_argument('--out', type=str, default="./PennFudanPed/PennFudanPed_prediction.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 