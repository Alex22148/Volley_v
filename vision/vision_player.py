import numpy as np
import cv2
import os

# ------------------------------------------------------------
# PARAMETRY
# ------------------------------------------------------------

WIDTH = 1920
HEIGHT = 1080
CAMERAS = 4
SCALE = 0.5

FRAME_SIZE = WIDTH * HEIGHT
QUAD_SIZE = 8 + CAMERAS * FRAME_SIZE  # timestamp + 4 obrazy


# ------------------------------------------------------------
# Klasa odczytu BIN (memmap)
# ------------------------------------------------------------

class VisionBinReader:

    def __init__(self, path):

        self.path = path
        self.filesize = os.path.getsize(path)

        self.num_frames = self.filesize // QUAD_SIZE

        print(f"Frames in file: {self.num_frames}")

        self.mm = np.memmap(
            path,
            dtype=np.uint8,
            mode='r'
        )

    # -----------------------------------------------------

    def get_frame(self, index):

        if index < 0 or index >= self.num_frames:
            return None

        offset = index * QUAD_SIZE

        # timestamp
        ts_bytes = self.mm[offset:offset + 8]
        timestamp = np.frombuffer(ts_bytes, dtype=np.uint64)[0]

        offset += 8

        frames = []

        for _ in range(CAMERAS):

            raw = self.mm[offset:offset + FRAME_SIZE]
            img = raw.reshape((HEIGHT, WIDTH))

            frames.append(img)

            offset += FRAME_SIZE

        return timestamp, frames


# ------------------------------------------------------------
# Quad preview
# ------------------------------------------------------------

def make_preview(frames):

    scaled = [
        cv2.resize(f, None, fx=SCALE, fy=SCALE)
        for f in frames
    ]

    top = np.hstack((scaled[0], scaled[1]))
    bottom = np.hstack((scaled[2], scaled[3]))

    return np.vstack((top, bottom))


# ------------------------------------------------------------
# PLAYER
# ------------------------------------------------------------

def main():

    path = r"C:\Users\Hyperbook\python_project\SAVE_ONLY\VolleyHub_shared_buffer_save_only_v3.7.0_bufor\buffer_recordings\multi_camera_buffer_20260317_093016\vision_dump_1773736216420372400.bin"

    reader = VisionBinReader(path)

    index = 0
    playing = False

    print("Controls:")
    print("a = back")
    print("d = forward")
    print("s = play")
    print("space = stop")
    print("ESC = exit")

    while True:

        if playing:
            index += 1
            if index >= reader.num_frames:
                index = reader.num_frames - 1
                playing = False

        data = reader.get_frame(index)

        if data is None:
            continue

        ts, frames = data

        preview = make_preview(frames)

        cv2.putText(
            preview,
            f"Frame: {index+1}",
            (750, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            2
        )

        cv2.imshow("Vision Player", preview)

        delay = 1 if playing else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break

        elif key == ord('a'):
            index = max(0, index - 1)

        elif key == ord('d'):
            index = min(reader.num_frames - 1, index + 1)

        elif key == ord('s'):
            playing = True

        elif key == ord(' '):
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()