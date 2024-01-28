import pandas as pd
from tqdm import tqdm
import cv2
import os
import glob


def load_data(filename):
    return pd.read_csv(
        filename,
        sep=",",
        header=None,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )


## Read all the images of the video sequence stored in ``folder``.
def read_images(folder):
    """
    Read all the images of the video sequence stored in ``folder``.
    """
    images = []
    for filename in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


def put_tracks_on_video(frames, df_det):
    # Iterate over the frames
    for frame in tqdm(range(len(frames))):
        # Get the detections of the current frame
        df_det_frame = df_det[df_det["frame"] == frame + 1]

        # Iterate over the detections
        for i in range(len(df_det_frame)):
            det = df_det_frame.iloc[i]
            # Draw the bounding box
            x, y, w, h = det[["bb_left", "bb_top", "bb_width", "bb_height"]].values
            cv2.rectangle(
                frames[frame],
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                2,
            )
            # Draw the ID
            cv2.putText(
                frames[frame],
                str(int(det["id"])),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    return frames


# Display video
def display_video(frames, fps=30):
    # Set frame size
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 720, 600)
    for frame in frames:
        cv2.imshow("frame", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def save_video(frames):
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter("MOT.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1080, 720))

    for frame in frames:
        out.write(frame)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
