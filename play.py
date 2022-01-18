#!python3

import argparse

from hammaren.input import show_input

parser = argparse.ArgumentParser(description="Play image/video.")

parser.add_argument(
    dest="input", type=str, help="Input: file or directory containing images/video."
)
parser.add_argument(
    '-m', dest="mode", type=str, help="Mode."
)

def main(args):
    show_input(args.input, args.mode)

if __name__ == "__main__":
    args = parser.parse_args()

    main(args)