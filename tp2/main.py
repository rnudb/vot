"""
 Load detections (det) stored in a MOT-challenge like formatted text file. Each line represents one object
instance and contains 10 values (fieldNames = [<frame>, <id>, <bb_left>, <bb_top>, <bb_width>,
<bb_height>, <conf>, <x>, <y>, <z>]
 frame = frame number
 id = number identifies that object as belonging to a trajectory by assigning a unique ID (set to
−1 in a detection file, as no ID is assigned yet).
 bb_left, bb_top, bb_width, bb_height: bounding box position in 2D image coordinates i.e. the
top-left corner as well as width and height
 conf: detection confidence score
 x,y,z: the world coordinates are ignored for the 2D challenge and can be filled with -1
"""

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_dt_meas = 0.1
y_dt_meas = 0.1


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


frames = read_images("../data/ADL-Rundle-6/img1")
print("Number of frames:", len(frames))


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


df_det = load_data("../data/ADL-Rundle-6/det/det.txt")
# # display histogram of confidence scores
# plt.hist(df_det['conf'], bins=100)
# plt.xlabel('Confidence score')
# plt.ylabel('Number of detections')
# plt.xticks(np.arange(df_det['conf'].min() - 1, df_det['conf'].max(), 10))
# plt.show()
# print("Estimation:", df_det.shape, "\n",df_det)
df_det = df_det[df_det["conf"] > 20]
df_det["kalman"] = None
print("Estimation:", df_det.shape, "\n", df_det)


df_gt = load_data("../data/ADL-Rundle-6/gt/gt.txt")
print("Ground Truth:", df_gt.shape, "\n", df_gt)


# Implement IoU for tracking
#  Compute similarity score using the Jaccard index (intersection-over-union) for each pair of
# bounding boxes
#  Create a similarity matrix that stores the IoU for all boxes
def compute_iou(bb1, bb2):
    """
    Computes the IoU of two bounding boxes.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Compute the area of both AABBs
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0
    return iou


def associate_detections_to_trackers(similarity_matrix, threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    A track gets the detection with the highest intersection-over-union to its last known object position
    (i.e. the previous detection of the track) assigned.
    """
    # Create a copy of the similarity matrix
    similarity_matrix = np.copy(similarity_matrix)

    # Create a list of matched detections between two frames
    matched_indices = []

    # Iterate over the similarity matrix
    while True:
        # Find the highest value in the similarity matrix
        max_value = np.max(similarity_matrix)
        if max_value < threshold:
            break

        # Find the indices of the highest value
        max_indices = np.where(similarity_matrix == max_value)

        # Get the first index
        max_row = max_indices[0][0]
        max_col = max_indices[1][0]

        # Add the matched indices to the list
        matched_indices.append((max_row, max_col))

        # Set the row and column of the matched indices to 0
        similarity_matrix[max_row] = 0
        similarity_matrix[:, max_col] = 0

    # Handle new tracks
    # Get the indices of the unmatched detections
    unmatched_detections = set(range(similarity_matrix.shape[0])).difference(
        [idx[0] for idx in matched_indices]
    )
    # Set the unmatched detections to new tracks
    for idx in unmatched_detections:
        matched_indices.append((idx, -1))

    # Handle lost tracks
    # Get the indices of the unmatched tracks
    unmatched_tracks = set(range(similarity_matrix.shape[1])).difference(
        [idx[1] for idx in matched_indices]
    )

    # Set the unmatched tracks to lost tracks (-1)
    for idx in unmatched_tracks:
        matched_indices.append((-1, idx))

    return matched_indices


## TP3
def associate_detections_to_trackers_hungarian(similarity_matrix, threshold=0.3):
    """
    Hungarian algorithm
    Assigns detections to tracked object (both represented as bounding boxes)
    A track gets the detection with the highest intersection-over-union to its last known object position
    (i.e. the previous detection of the track) assigned.
    """
    # Create a copy of the similarity matrix
    similarity_matrix = np.copy(similarity_matrix)

    # Match detections using the Hungarian algorithm
    matched_indices = linear_sum_assignment(similarity_matrix, maximize=True)

    # Create a list of matched detections between two frames
    matched_indices = [
        (matched_indices[0][i], matched_indices[1][i])
        for i in range(len(matched_indices[0]))
    ]

    # Handle new tracks
    # Get the indices of the unmatched detections
    unmatched_detections = set(range(similarity_matrix.shape[0])).difference(
        [idx[0] for idx in matched_indices]
    )
    # Set the unmatched detections to new tracks
    for idx in unmatched_detections:
        matched_indices.append((idx, -1))

    # Handle lost tracks
    # Get the indices of the unmatched tracks
    unmatched_tracks = set(range(similarity_matrix.shape[1])).difference(
        [idx[1] for idx in matched_indices]
    )

    # Set the unmatched tracks to lost tracks (-1)
    for idx in unmatched_tracks:
        matched_indices.append((-1, idx))

    return matched_indices


def update_tracks_association(df_det, matching, frame):
    for idx in matching:
        if idx[0] != -1:
            if idx[1] == -1:
                # New track
                df_det.at[df_det[df_det["frame"] == frame].iloc[idx[0]].name, "id"] = (
                    df_det["id"].max() + 1
                )
            else:
                # Update track
                df_det.at[
                    df_det[df_det["frame"] == frame].iloc[idx[0]].name, "id"
                ] = df_det[df_det["frame"] == frame - 1].iloc[idx[1]]["id"]


def compute_tracks_association_greedy(df_det):
    """ """
    df_det = df_det.sort_values(by=["frame"])
    max_frame = df_det["frame"].max()
    matching = []

    # Init first frame
    df_det_frame = df_det[df_det["frame"] == 1]
    for i in range(len(df_det_frame)):
        df_det.at[df_det_frame.iloc[i].name, "id"] = i + 1

    # Iterate over the frames
    for frame in tqdm(range(2, max_frame)):
        # Get the detections of the current frame
        df_det_frame = df_det[df_det["frame"] == frame]
        # Get the detections of the previous frame
        df_det_frame_prev = df_det[df_det["frame"] == frame - 1]

        # print(df_det_frame.shape, df_det_frame_prev.shape)

        # Compute the similarity matrix
        similarity_matrix = np.zeros(
            (df_det_frame.shape[0], df_det_frame_prev.shape[0])
        )
        for i in range(len(df_det_frame)):
            det = df_det_frame.iloc[i]
            for j in range(len(df_det_frame_prev)):
                det_prev = df_det_frame_prev.iloc[j]
                similarity_matrix[i, j] = compute_iou(
                    det[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                    det_prev[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                )

        # Associate the detections to tracks
        # matched_indices = associate_detections_to_trackers(similarity_matrix, threshold=0.1)
        matched_indices = associate_detections_to_trackers_hungarian(
            similarity_matrix, threshold=0.1
        )
        matching.append(matched_indices)

        # Update the tracks
        update_tracks_association(df_det, matching[-1], frame)

    return matching, df_det


# print("==== Compute track association ====")
# res, df_det = compute_tracks_association_greedy(df_det)
# print(df_det)


def compute_bbox_to_centroid(row):
    """ """
    return row["bb_left"] + row["bb_width"] / 2, row["bb_top"] + row["bb_height"] / 2


def update_kalman_filters(df_det, frame_idx):
    """
    Update or create the kalman filter for the tracks of the current frame
    """
    # Get the detections of the current frame
    df_det_frame = df_det.loc[df_det["frame"] == frame_idx]

    # Iterate over the detections
    for i in range(len(df_det_frame)):
        det = df_det_frame.iloc[i]
        # Create a new kalman filter if the track does not have one
        if det["kalman"] is None:
            det.loc["kalman"] = KalmanFilter(
                dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas
            )
            c_x, c_y = compute_bbox_to_centroid(det)
            det["kalman"].x_k = np.array([[c_x], [c_y], [0], [0]])

            # Predict the state
            det["kalman"].predict()
        else:
            # Update the kalman filter
            det["kalman"].predict()
            c_x, c_y = compute_bbox_to_centroid(det)
            det["kalman"].update(np.array([[c_x], [c_y]]))

        # Update the rows positions
        df_det.at[det.name, "bb_left"] = int(det["kalman"].x_k[0] - det["bb_width"] / 2)
        df_det.at[det.name, "bb_top"] = int(det["kalman"].x_k[1] - det["bb_height"] / 2)

    return df_det


## TP4
def update_tracks_association_kalman(df_det, matching, frame):
    for idx in matching:
        if idx[0] != -1:
            if idx[1] == -1:
                # New track
                df_det.at[df_det[df_det["frame"] == frame].iloc[idx[0]].name, "id"] = (
                    df_det["id"].max() + 1
                )
            else:
                # Update track
                df_det.at[
                    df_det[df_det["frame"] == frame].iloc[idx[0]].name, "id"
                ] = df_det[df_det["frame"] == frame - 1].iloc[idx[1]]["id"]
                df_det.at[
                    df_det[df_det["frame"] == frame].iloc[idx[0]].name, "kalman"
                ] = df_det[df_det["frame"] == frame - 1].iloc[idx[1]][
                    "kalman"
                ]  # TP4 -> keep same kalman filter for the track


def compute_tracks_association_kalman(df_det):
    df_det = df_det.sort_values(by=["frame"])
    max_frame = df_det["frame"].max()
    matching = []

    # Init first frame
    df_det_frame = df_det[df_det["frame"] == 1]
    for i in range(len(df_det_frame)):
        df_det.at[df_det_frame.iloc[i].name, "id"] = i + 1

    update_kalman_filters(df_det, 1)

    # Iterate over the frames
    for frame in tqdm(range(2, max_frame)):
        # Get the detections of the current frame
        df_det_frame = df_det[df_det["frame"] == frame]
        # Get the detections of the previous frame
        df_det_frame_prev = df_det[df_det["frame"] == frame - 1]

        # print(df_det_frame.shape, df_det_frame_prev.shape)

        # Compute the similarity matrix
        similarity_matrix = np.zeros(
            (df_det_frame.shape[0], df_det_frame_prev.shape[0])
        )
        for i in range(len(df_det_frame)):
            det = df_det_frame.iloc[i]
            for j in range(len(df_det_frame_prev)):
                det_prev = df_det_frame_prev.iloc[j]
                similarity_matrix[i, j] = compute_iou(
                    det[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                    det_prev[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                )

        # Associate the detections to tracks
        # matched_indices = associate_detections_to_trackers(similarity_matrix, threshold=0.1)
        matched_indices = associate_detections_to_trackers_hungarian(
            similarity_matrix, threshold=0.1
        )
        matching.append(matched_indices)

        # Update the tracks
        update_tracks_association_kalman(df_det, matching[-1], frame)

        # Update the kalman filters
        df_det = update_kalman_filters(df_det, frame)

    return matching, df_det


print("==== Compute track association ====")
res, df_det = compute_tracks_association_kalman(df_det)
print(df_det)


def put_tracks_on_video(frames, df_det):
    """ """
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
def display_video(frames):
    for frame in frames:
        cv2.imshow("frame", frame)
        if cv2.waitKey(int(1000 / 10)) & 0xFF == ord("q"):
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


frames = put_tracks_on_video(frames, df_det)
display_video(frames)
save_video(frames)
