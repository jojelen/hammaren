#!python3

import argparse

from hammaren.input import show_input
from hammaren.lane_detection import detect_lanes

parser = argparse.ArgumentParser(description="Play image/video.")

parser.add_argument(
    "-i",
    dest="input",
    type=str,
    help="Input: file or directory containing images/video. Video device 0 will be used if not specified.",
)
parser.add_argument("--tflite", type=str, help="Tflite model file path.")
parser.add_argument("--lane-detection", action="store_true", default=False, help="Run lane detection algorithm.")
parser.add_argument("-l", "--loop", action="store_true", default=False, help="Loop input.")



def main(args):
    algorithms = []
    if args.lane_detection:
        algorithms.append(detect_lanes)
    show_input(args.input, args.tflite, algorithms, args.loop)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
