import os
import cv2

from .detectron import run_detectron

def get_images_in_folder(folder):
    for f in os.scandir(folder):
        if not f.is_file():
            continue

        image = cv2.imread(os.path.join(folder, f.name))

        yield f.name, image

def get_frames_in_video(video):
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield "frame", frame

def get_frames_in_input(path):
    if os.path.isdir(path):
        it = get_images_in_folder(path)
    else:
        it = get_frames_in_video(path)
    return it

def resize_to_width(width, image):
    h, w, _ = image.shape
    scale = w / float(width)
    w = int(w / scale)
    h = int(h / scale)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

def show_input(path):
    it = get_frames_in_input(path)

    for name, image in it:
        image = resize_to_width(1280, image)
        cv2.imshow(name, image)

        run_detectron(image)

        # Quit or paus.
        key = cv2.waitKey(10)
        if  key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("p"):
            while True:
                if cv2.waitKey(10) & 0xFF == ord("p"):
                    break
