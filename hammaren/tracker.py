global_track_id = 0

class Tracker:
    def __init__(self):
        self.tracks = []
        self._high_score_threshold = 0.5

    def add_detections(self, objects):
        high_score_objs = [o for o in objects if o.score > self._high_score_threshold]
        pass


class Track:
    def __init__(self, class_id, bbox):
        global global_track_id

        self._track_id = global_track_id
        global_track_id += 1

        self._class_id = class_id
        self._num_undetected_frames = 0




