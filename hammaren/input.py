import os
import cv2

from .fps import FPSCounter
from .utils import draw_objects
from .tracker import Tracker

# from .detectron import run_detectron


def get_images_in_folder(folder):
    for f in os.scandir(folder):
        if not f.is_file():
            continue

        image = cv2.imread(os.path.join(folder, f.name))

        yield f.name, image


def get_frames_in_video(video):
    cap = cv2.VideoCapture(video)
    if video == 0:
        print("Setting capture size!")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield "frame", frame


def get_frames_in_input(path):
    if path is None:
        it = get_frames_in_video(0)
    elif os.path.isdir(path):
        it = get_images_in_folder(path)
    elif os.path.isfile(path):
        it = get_frames_in_video(path)
    return it


def resize_to_width(width, image):
    h, w, _ = image.shape
    scale = w / float(width)
    w = int(w / scale)
    h = int(h / scale)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_image(image, size):
    h, w, _ = image.shape
    w_pad = int((w - size[0]) / 2.0)
    h_pad = int((h - size[1]) / 2.0)

    return image[h_pad : h_pad + size[1], w_pad : w_pad + size[0]]


def make_square(image, center=False):
    h, w, _ = image.shape
    smallest = h if h < w else w

    if center:
        return crop_image(image, [smallest, smallest])
    else:
        return image[:smallest,:smallest]


def show_input(path, tflite_model, algorithms, loop):

    if tflite_model:
        from .tflite import TFLiteModel
        model = TFLiteModel(tflite_model)
        input_size = model.input_size
        labels = model.labels

    tracker = Tracker()


    while True:
        it = get_frames_in_input(path)
        fps_counter = FPSCounter()
        for name, image in it:
            for algo in algorithms:
                algo(image)
            if tflite_model:
                # TODO: Remove
                image = make_square(image)

                rgb_image = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                objects = model.run_inference(rgb_image, score_threshold=0.5)

                tracker.add_detections(objects)

                image = draw_objects(image, objects, input_size, labels)

            image = resize_to_width(1024, image)
            image = fps_counter.draw_fps(image)
            cv2.imshow(name, image)

            # Quit or paus.
            key = cv2.waitKey(10)
            if key & 0xFF == ord("q"):
                return
            elif key & 0xFF == ord("p"):
                while True:
                    if cv2.waitKey(10) & 0xFF == ord("p"):
                        break
            fps_counter.end_frame()

        if not loop:
            return
