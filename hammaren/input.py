import os
import cv2

from .tflite import TFLiteModel

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


def show_input(path, mode, tflite_model):
    it = get_frames_in_input(path)

    model = TFLiteModel(tflite_model)
    input_size = model.input_size

    for name, image in it:
        if mode == "detectron2":
            raise NotImplementedError()
            # run_detectron(image)

        image = crop_image(image, input_size)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model.run_inference(rgb_image)
        image = model.draw_results(image)

        image = resize_to_width(1280, image)
        cv2.imshow(name, image)

        # Quit or paus.
        key = cv2.waitKey(10)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("p"):
            while True:
                if cv2.waitKey(10) & 0xFF == ord("p"):
                    break
