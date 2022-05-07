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

        self.labels = read_label_file("models/coco-labels.txt")
        self.input_size = input_size(self._interpreter)

    def run_inference(self, image, score_threshold=0.1):
        """
        Run inference on image and return list of object detections.
        """
        edgetpu.run_inference(self._interpreter, image.tobytes())

        return get_objects(self._interpreter, score_threshold)
