import torch
import argparse

def main(model_path):

    model = torch.load(model_path)

    print(model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('path')

    args = parser.parse_args()

    main(args.path)