import cv2

from pycoral.utils import edgetpu
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file


def print_tensor(tensor):
    print("{}: {}, dtype={}".format(tensor["name"], tensor["shape"], tensor["dtype"]))


def print_tensors(tensors):
    for tensor in tensors:
        print_tensor(tensor)


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = "{}% {}".format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(
            cv2_im, label, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2
        )
    return cv2_im


def load_tflite_model(model_path):
    tpus = edgetpu.list_edge_tpus()
    if tpus is None:
        raise RuntimeError("No EdgeTPU device found.")
    print("Found EdgeTPU device: ", tpus)
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    print("Tflite model loaded!")

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    print("Input tensors:")
    print_tensors(inputs)
    print_tensors(outputs)

    return interpreter


class TFLiteModel:
    def __init__(self, model_path):
        self._interpreter = load_tflite_model(model_path)
        self._threshold = 0.5
        self._labels = read_label_file("models/coco-labels.txt")
        self.input_size = input_size(self._interpreter)

    def run_inference(self, image):
        edgetpu.run_inference(self._interpreter, image.tobytes())

    def draw_results(self, image):
        objs = get_objects(self._interpreter, self._threshold)
        image = append_objs_to_img(image, self.input_size, objs, self._labels)
        return image
