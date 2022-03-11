#!python3

import argparse

from hammaren.input import show_input

parser = argparse.ArgumentParser(description="Play image/video.")

parser.add_argument(
    "-i",
    dest="input",
    type=str,
    help="Input: file or directory containing images/video. Video device 0 will be used if not specified.",
)
parser.add_argument("-m", dest="mode", type=str, help="Mode.")
parser.add_argument("--tflite", type=str, help="Tflite model file path.")


def main(args):
    show_input(args.input, args.mode, args.tflite)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
