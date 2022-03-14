import time

class FPSCounter:
    def __init__(self, fps_cap = 30):
        self._last_frame_time = None
        self._curr_frame = 0
        self._max_frames = 10
        self._frame_times = [0] * self._max_frames
        self._fps_cap = fps_cap
        self._cap_frame_time = None if not fps_cap else 1e9 / float(fps_cap)


    def end_frame(self):
        if not self._last_frame_time:
            self._last_frame_time = time.time_ns()
            return

        frame_time = time.time_ns() - self._last_frame_time
        if self._cap_frame_time and frame_time < self._cap_frame_time:
            sleep_ns = self._cap_frame_time - frame_time
            time.sleep(sleep_ns * 1e-9)
            self._frame_times[self._curr_frame]  = time.time_ns() - self._last_frame_time
        else:
            self._frame_times[self._curr_frame]  = frame_time

        # Increment current-frame counter.
        self._curr_frame += 1
        if self._curr_frame >= self._max_frames:
            self._curr_frame = 0

        self._last_frame_time = time.time_ns()

    def _calc_fps(self):
        average_frame_time = sum(self._frame_times) / self._max_frames
        if average_frame_time == 0:
            self._fps = 0
        else:
            self._fps = 1e9 / float(average_frame_time)

    def draw_fps(self, image):
        self._calc_fps()
        print("FPS: ", self._fps)

        return image

