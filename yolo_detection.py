
import torch
import cv2
import pandas as pd

import glob
import os

from tqdm import tqdm


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



def convert_yolo_format_to_bbox(xmin, ymin, xmax, ymax, confidence):
    return [int(xmin.item()), int(ymin.item()), (xmax - xmin).item(), (ymax - ymin).item(), confidence.item() * 100]


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')

DATAPATH = "data/MOT15/train"

# Walk through all the folders in the data folder and redo the detections
for folder in os.listdir(DATAPATH):
    print("Processing folder: " + folder)

    # Get the path to the images
    img_path = os.path.join(DATAPATH, folder, "img1")

    images = read_images(img_path)

    # Iterate over the frames
    detections = []
    for frame in tqdm(range(len(images))):
        # Get the current frame
        img = images[frame]

        # Convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform the detection
        results = model(img)

        # Get only the persons
        persons = results.xyxy[0][results.xyxy[0][:, -1] == 0]

        # print(persons)

        # Iterate over the persons
        for person in persons:
            detections.append([frame + 1, -1, *convert_yolo_format_to_bbox(*person[:5]), -1, -1, -1])

    # Create the dataframe
    df_det = pd.DataFrame(
        detections,
        columns=[
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

    df_det = df_det.astype({"frame": int, "id": int, "bb_left": int, "bb_top": int, "bb_width": float, "bb_height": float, "conf": float, "x": int, "y": int, "z": int})

    # Create the folder if needed
    if not os.path.exists("data/yolo_dets/" + folder + "/det"):
        os.makedirs("data/yolo_dets/" + folder + "/det")


    # Save the dataframe
    df_det.to_csv("data/yolo_dets/" + folder + "/det/det.txt", index=False, sep=',', header=False)