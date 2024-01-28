import cv2
import torch
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

preprocess = (
    models.ResNet18_Weights.IMAGENET1K_V1.transforms()
)  # Use the last version of the transforms

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Add to the model a last layer that outputs a 512-dimensional vector
model.fc = torch.nn.Linear(model.fc.in_features, 512)
model.eval()


class ImageEmbedder:
    def __init__(self, df_det, image_folder, use_color_histogram=False):
        self.df_det = df_det
        self.image_folder = image_folder
        self.use_color_histogram = use_color_histogram

        self.frames = []
        self.model = model

        self.preprocess = preprocess

        self.distance = self.compute_cosine_similarity

        self.max_distance = np.inf

    def load_images(self):
        print("Loading images...")
        for i in tqdm(range(1, max(self.df_det.frame.unique()) + 1)):
            img = cv2.imread(f"{self.image_folder}/{str(i).zfill(6)}.jpg")
            if img is not None:
                self.frames.append(img)
            else:
                print(f"Image {i} not found.")
        print("Done.")

    def make_histogram(self, bbox, frame):
        """
        Computes rgb histogram for a given bbox
        """
        # Get the bbox
        x, y, w, h = bbox

        # Get the image
        img = self.frames[frame - 1]

        # Crop the image
        cropped = img[int(y) : int(y + h), int(x) : int(x + w)]

        # Compute the histogram
        hist = cv2.calcHist(
            [cropped], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def compute_color_histograms(self):
        print("Computing color histograms...")

        batch_size = 256

        histograms = []
        # Compute embeddings in batches
        print(f"Computing histograms in batches of {batch_size} ...")
        for i in tqdm(range(0, len(self.df_det), batch_size)):
            # Get all bboxes from the dataframe
            bboxes = self.df_det[["bb_left", "bb_top", "bb_width", "bb_height"]].values[
                i : min(i + batch_size, len(self.df_det))
            ]

            # make histogram for the given bbox
            for el in [
                self.make_histogram(bbox, frame)
                for bbox, frame in zip(
                    bboxes,
                    self.df_det.frame.values[i : min(i + batch_size, len(self.df_det))],
                )
            ]:
                histograms.append(torch.Tensor(list(el)))

        self.df_det["color_histograms"] = histograms

        print("Done.")

    def compute_embeddings(self):
        print("Computing embeddings...")
        # Make batches of images
        batch_size = 128

        embeddings = []
        # Compute embeddings in batches
        print(f"Computing embeddings in batches of {batch_size} ...")
        for i in tqdm(range(0, len(self.df_det), batch_size)):
            # Get all bboxes from the dataframe
            bboxes = self.df_det[["bb_left", "bb_top", "bb_width", "bb_height"]].values[
                i : min(i + batch_size, len(self.df_det))
            ]

            # Get all matching image per row of the dataframe
            images = [
                self.frames[i - 1]
                for i in self.df_det.frame.values[
                    i : min(i + batch_size, len(self.df_det))
                ]
            ]

            # Crop all images
            cropped = [
                img[int(y) : int(y + h), int(x) : int(x + w)]
                for (x, y, w, h), img in zip(bboxes, images)
            ]

            # Resize all images to 224x224
            resized = [cv2.resize(img, (224, 224)) for img in cropped]

            # Convert to PIL images
            images = [transforms.ToPILImage()(img) for img in resized]

            # Convert to tensors
            tensors = [self.preprocess(img) for img in images]

            with torch.no_grad():
                batch_embeddings = self.model(torch.stack(tensors))
                batch_embeddings = batch_embeddings.squeeze()
                batch_embeddings = batch_embeddings / torch.norm(
                    batch_embeddings, dim=1, keepdim=True
                )
                # batch_embeddings = batch_embeddings.unsqueeze(dim=1)
                for i in range(batch_embeddings.shape[0]):
                    embeddings.append(batch_embeddings[i])

        self.df_det["embedding"] = embeddings
        print("Done.")

    # Cosine similarity because we have high dimensional vectors and we already have IoUs that range between 0 and 1
    # As cosine is between -1 and 1, we can use it as a distance metric that works well with the IoU
    def compute_cosine_similarity(self, embedding1, embedding2):
        return torch.abs(
            torch.nn.CosineSimilarity(dim=0)(embedding1, embedding2)
        ).item()

    # Euclidean distance because we have high dimensional vectors could also work
    def compute_euclidean_distance(self, embedding1, embedding2):
        return torch.dist(embedding1, embedding2).item()
