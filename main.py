import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pedestrian Detection')
    
    parser.add_argument('-i', '--data_dir', type=str, default='Liquor', required=True, help="Path to the dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # if args.method not in [1,2,3]:
    #     raise ValueError("method should be one of 1/2/3 - Found: %u"%args.method)
    # FUNCTION_MAPPER = {
    #         1: appearence_based_tracking,
    #         2: lucas_kanade,
    #         3: pyramid_lk,
    #     }

    # print("using %s for template tracking"%(FUNCTION_MAPPER[args.method]))
    # FUNCTION_MAPPER[args.method](args)

    # if args.eval:
    #     print("evaluating...")
    #     subprocess.call("python eval.py -g %s/groundtruth_rect.txt -p %s/predictions_part%s.txt"%(args.data_dir, args.data_dir, args.method), shell=True)