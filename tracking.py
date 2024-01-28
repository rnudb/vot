import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from scipy.optimize import linear_sum_assignment

from image_embedder import ImageEmbedder
from kalman_filter import KalmanFilter

import torch

image_embedder = ImageEmbedder(None, None)


class MultiObjectTracking:
    def __init__(
        self,
        df_det,
        threshold_iou=0.2,
        threshold_conf=None,
        use_hungarian_matching=False,
        use_kalman_filters=False,
        use_embeddings=False,
        use_color_histogram=False,
        image_folder="data/ADL-Rundle-6/img1",
    ) -> None:
        self.use_hungarian_matching = use_hungarian_matching
        self.use_kalman_filters = use_kalman_filters
        self.use_embeddings = use_embeddings
        self.use_color_histogram = use_color_histogram
        self.threshold_iou = threshold_iou
        self.image_folder = image_folder
        self.image_embedder = None

        # Add kalman filters to the dataframe
        df_det["kalman"] = None
        self.df_det = df_det.sort_values(by=["frame"])

        # Filter out low confidence detections
        if threshold_conf is not None:
            self.df_det = self.df_det[self.df_det["conf"] > threshold_conf]

        # Set the confidence to 1, to match the format of the MOT15 dataset
        self.df_det["conf"] = 1

        # Clamp the bounding boxes to the image size
        self.df_det["bb_left"] = self.df_det["bb_left"].clip(lower=0, upper=1920)
        self.df_det["bb_top"] = self.df_det["bb_top"].clip(lower=0, upper=1080)

        # Remove < 5 width or height detections
        self.df_det = self.df_det[
            (self.df_det["bb_width"] > 5) & (self.df_det["bb_height"] > 5)
        ]

        # Set the default kalman parameters
        self.default_kalman_parameters = {
            "dt": 1,  # Time step
            "u_x": 0,  # Acceleration in x direction
            "u_y": 0,  # Acceleration in y direction
            "std_acc": 0.1,  # Standard deviation of the acceleration
            "x_dt_meas": 1,  # Measurement noise in x direction
            "y_dt_meas": 1,  # Measurement noise in y direction
        }

        self.df_det["embedding"] = None
        self.df_det["color_histograms"] = None
        if self.use_embeddings or self.use_color_histogram:
            # Update image embdder
            self.image_embedder = image_embedder
            self.image_embedder.df_det = self.df_det
            self.image_embedder.image_folder = image_folder
            self.image_embedder.use_color_histogram = use_color_histogram
            self.image_embedder.frames = []
            self.image_embedder.load_images()

            if use_embeddings:
                self.image_embedder.compute_embeddings()
                self.df_det["embedding"] = self.image_embedder.df_det["embedding"]

            if use_color_histogram:
                self.image_embedder.compute_color_histograms()
                self.df_det["color_histograms"] = self.image_embedder.df_det[
                    "color_histograms"
                ]

    """
        UTILS
    """

    def _compute_bbox_to_centroid(self, row):
        return (
            row["bb_left"] + row["bb_width"] / 2,
            row["bb_top"] + row["bb_height"] / 2,
        )

    def compute_iou(self, bb1, bb2):
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

    def update_kalman_filters(self, frame_idx):
        """
        Update or create the kalman filter for the tracks of the current frame
        """
        # Get the detections of the current frame
        df_det_frame = self.df_det.loc[self.df_det["frame"] == frame_idx]

        # Iterate over the detections
        for i in range(len(df_det_frame)):
            det = df_det_frame.iloc[i]
            # Create a new kalman filter if the track does not have one
            if det["kalman"] is None:
                det.loc["kalman"] = KalmanFilter(
                    *self.default_kalman_parameters.values()
                )
                c_x, c_y = self._compute_bbox_to_centroid(det)
                det["kalman"].x_k = np.array([[c_x], [c_y], [0], [0]])

                # Predict the state
                det["kalman"].predict()
            else:
                # Update the kalman filter
                det["kalman"].predict()
                c_x, c_y = self._compute_bbox_to_centroid(det)
                det["kalman"].update(np.array([[c_x], [c_y]]))

            # Update the rows positions
            self.df_det.at[det.name, "kalman"] = det["kalman"]

    """
        MAIN LOGIC
    """

    def first_frame_track_association(self):
        # Init first frame
        df_det_frame = self.df_det[self.df_det["frame"] == 1]
        for i in range(len(df_det_frame)):
            self.df_det.at[df_det_frame.iloc[i].name, "id"] = i + 1

        if self.use_kalman_filters:
            self.update_kalman_filters(1)

    def compute_similarity_matrix(self, frame):
        # Get the detections of the current frame
        df_det_frame = self.df_det.loc[self.df_det["frame"] == frame]
        # Get the detections of the previous frame
        df_det_frame_prev = self.df_det.loc[self.df_det["frame"] == frame - 1]

        # Init the similarity matrices
        similarity_matrix_IoU = np.zeros(
            (df_det_frame.shape[0], df_det_frame_prev.shape[0])
        )
        similarity_matrix_embedding = np.zeros(
            (df_det_frame.shape[0], df_det_frame_prev.shape[0])
        )
        similarity_matrix_histogram = np.zeros(
            (df_det_frame.shape[0], df_det_frame_prev.shape[0])
        )

        for i in range(len(df_det_frame)):
            # Get the detection
            det = df_det_frame.iloc[i]
            for j in range(len(df_det_frame_prev)):
                # Get the previous detection
                det_prev = df_det_frame_prev.iloc[j]

                if self.use_kalman_filters:
                    # Update det_prev pos with the expected new pos from kalman
                    det_prev["bb_left"] = int(
                        det_prev["kalman"].x_k[0] - det_prev["bb_width"] / 2
                    )
                    det_prev["bb_top"] = int(
                        det_prev["kalman"].x_k[1] - det_prev["bb_height"] / 2
                    )

                # Compute the similarity between the two detections
                similarity_matrix_IoU[i, j] += self.compute_iou(
                    det[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                    det_prev[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
                )

                if self.use_embeddings:
                    similarity_matrix_embedding[i, j] += self.image_embedder.distance(
                        det["embedding"], det_prev["embedding"]
                    )

                if self.use_color_histogram:
                    dist = self.image_embedder.distance(
                        det["color_histograms"], det_prev["color_histograms"]
                    )
                    similarity_matrix_histogram[i, j] += dist

        # Normalize the similarity matrices
        similarity_matrix_IoU /= (
            similarity_matrix_IoU.max() if similarity_matrix_IoU.max() != 0 else 1
        )
        similarity_matrix_embedding /= (
            similarity_matrix_embedding.max()
            if similarity_matrix_embedding.max() != 0
            else 1
        )
        similarity_matrix_histogram /= (
            similarity_matrix_histogram.max()
            if similarity_matrix_histogram.max() != 0
            else 1
        )

        # Compute the final similarity matrix
        similarity_matrix = similarity_matrix_IoU
        if self.use_embeddings:
            similarity_matrix += similarity_matrix_embedding
        if self.use_color_histogram:
            similarity_matrix += similarity_matrix_histogram

        # Normalize the similarity matrix (useful for thresholding)
        if not self.use_hungarian_matching:
            similarity_matrix /= (
                similarity_matrix.max() if similarity_matrix.max() != 0 else 1
            )

        return similarity_matrix

    def _associate_detection_new_tracks(self, similarity_matrix, matched_indices):
        # Get the indices of the unmatched detections
        unmatched_detections = set(range(similarity_matrix.shape[0])).difference(
            [idx[0] for idx in matched_indices]
        )
        # Set the unmatched detections to new tracks
        for idx in unmatched_detections:
            matched_indices.append((idx, -1))

        return matched_indices

    def _associate_detection_lost_tracks(self, similarity_matrix, matched_indices):
        # Get the indices of the unmatched tracks
        unmatched_tracks = set(range(similarity_matrix.shape[1])).difference(
            [idx[1] for idx in matched_indices]
        )

        # Set the unmatched tracks to lost tracks (-1)
        for idx in unmatched_tracks:
            matched_indices.append((-1, idx))

        return matched_indices

    def associate_detections_to_trackers(self, similarity_matrix):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        A track gets the detection with the highest intersection-over-union to its last known object position
        (i.e. the previous detection of the track) assigned.
        """
        # Create a copy of the similarity matrix
        similarity_matrix = np.copy(similarity_matrix)

        # Create a list of matched detections between two frames
        matched_indices = []

        if self.use_hungarian_matching:
            # Match detections using the Hungarian algorithm
            matched_indices = linear_sum_assignment(similarity_matrix, maximize=True)

            # Create a list of matched detections between two frames
            matched_indices = [
                (matched_indices[0][i], matched_indices[1][i])
                for i in range(len(matched_indices[0]))
            ]
        else:
            # Iterate over the similarity matrix
            while True:
                # Find the highest value in the similarity matrix
                max_value = np.max(similarity_matrix)
                if max_value < self.threshold_iou:
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
        matched_indices = self._associate_detection_new_tracks(
            similarity_matrix, matched_indices
        )

        # Handle lost tracks
        matched_indices = self._associate_detection_lost_tracks(
            similarity_matrix, matched_indices
        )

        return matched_indices

    def update_tracks_association(self, matching, frame):
        for idx in matching:
            if idx[0] == -1:
                continue  # Skip lost track

            if idx[1] == -1:  # New track
                self.df_det.at[
                    self.df_det[self.df_det["frame"] == frame].iloc[idx[0]].name, "id"
                ] = (self.df_det["id"].max() + 1)
            else:  # Update existing track
                self.df_det.at[
                    self.df_det[self.df_det["frame"] == frame].iloc[idx[0]].name, "id"
                ] = self.df_det[self.df_det["frame"] == frame - 1].iloc[idx[1]]["id"]

                if self.use_kalman_filters:
                    self.df_det.at[
                        self.df_det[self.df_det["frame"] == frame].iloc[idx[0]].name,
                        "kalman",
                    ] = self.df_det[self.df_det["frame"] == frame - 1].iloc[idx[1]][
                        "kalman"
                    ]

    def compute_tracks_association(self):
        """
        Main Loop
        """
        max_frame = self.df_det["frame"].max()
        matching = []

        self.first_frame_track_association()

        print("Computing tracks association...")

        # Iterate over the frames
        for frame in tqdm(range(2, max_frame + 1)):
            df_det_frame = self.df_det.loc[self.df_det["frame"] == frame]

            # If there are no detections, skip the frame
            if len(df_det_frame) == 0:
                continue

            # Get the detections of the previous frame
            df_det_frame_prev = self.df_det.loc[self.df_det["frame"] == frame - 1]

            # If there are no detections in the previous frame, init the current frame to new tracks
            if len(df_det_frame_prev) == 0:
                current_max_id = self.df_det["id"].max()
                if current_max_id == -1:
                    current_max_id = 0

                for i in range(len(df_det_frame)):
                    self.df_det.at[df_det_frame.iloc[i].name, "id"] = (
                        current_max_id + i + 1
                    )

                if self.use_kalman_filters:
                    self.update_kalman_filters(frame)

                continue

            # Compute similarity matrix beetween the previous tracks and the current detections
            similarity_matrix = self.compute_similarity_matrix(frame)

            # Associate the detections to tracks
            matched_indices = self.associate_detections_to_trackers(similarity_matrix)
            matching.append(matched_indices)  # Store the matching for the current frame

            # Update the tracks
            self.update_tracks_association(matching[-1], frame)

            # Update kalman filters
            if self.use_kalman_filters:
                self.update_kalman_filters(frame)

        print("Done.")
