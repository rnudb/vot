import pandas as pd
from utils import load_data, read_images, put_tracks_on_video, display_video
from tracking import MultiObjectTracking

import os
import time

pd.options.mode.chained_assignment = None  # default='warn'

DATA_FOLDER = "data/MOT15/train"
OLD_DATA_FOLDER = None

# Check if there are custom detections made with YOLO
if len(os.listdir("data/yolo_dets")) == len(os.listdir(DATA_FOLDER)):
    OLD_DATA_FOLDER = DATA_FOLDER
    DATA_FOLDER = "data/yolo_dets"

def process_sequence(folder):
    print("Processing folder: " + folder)

    # Get the path to the det file
    det_file = os.path.join(DATA_FOLDER, folder, "det/det.txt")

    # Load the det file
    df_det = load_data(det_file)

    # Create our MultiObjectTracking object
    mot = MultiObjectTracking(
        df_det,
        use_hungarian_matching=True,
        use_kalman_filters=True,
        use_embeddings=True,
        use_color_histogram=True,
        threshold_iou=0.8,
        threshold_conf=int(df_det["conf"].describe()["25%"]), # remove worst 25%
        image_folder=os.path.join("data/MOT15/train", folder, "img1"),
    )

    # Compute tracks association
    mot.compute_tracks_association()

    # Save the associated tracks in a file
    mot.df_det.iloc[:, list(range(10))].to_csv("data/outputs/" + folder + ".txt", index=False, header=False, sep=",")

    # Print results' summary
    print("Final number of tracks: " + str(mot.df_det["id"].max()))

def main():

    processing_times = {}

    for folder in os.listdir(DATA_FOLDER):
        # Skip the folder if it's not a directory
        if not os.path.isdir(os.path.join(DATA_FOLDER, folder)):
            continue

        if folder != "ADL-Rundle-6":
            continue

        # Measure the start time
        start_time = time.time()

        # Process the sequence
        process_sequence(folder)

        # Measure the end time
        end_time = time.time()

        # Compute the processing time
        processing_time = end_time - start_time

        # Compute the frames per second (fps)
        fps = len(os.listdir(os.path.join(OLD_DATA_FOLDER if OLD_DATA_FOLDER is not None else DATA_FOLDER, folder, "img1"))) / processing_time

        # Store the processing time and fps in the dictionary
        processing_times[folder] = (processing_time, fps)

    # Save the processing times to a file
    with open("processing_times.txt", "w") as file:
        for folder, (processing_time, fps) in processing_times.items():
            file.write(f"{folder}: {processing_time} seconds, {fps} fps\n")

if __name__ == "__main__":
    main()
