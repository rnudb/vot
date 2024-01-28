## Repo for my project of object tracking.

## Setup

Make sure to have installed the virtual environment with the requirements.txt file and have activated it.

To download the dataset, please get the MOT15 dataset from [here](https://motchallenge.net/data/MOT15/). Then, put the MOT15 dataset in the `data` folder. (under the name "MOT15")

Please also add a `outputs` folder in the `data` folder. This is where the detections with associated scores will be stored.
Finally, add a `yolo_dets` folder in the `data` folder. This is where the detections from YOLO will be stored. It should be in the same format as the MOT15 dataset. (only the train folder of MOT15 is needed as it is the only one used)

The folder structure should look like this:

```
data
└── MOT15
    ├── test
    │   ├── ADL-Rundle-6
    │   ├── ETH-Bahnhof
    │   ├── ETH-Pedcross2
    │   ├── KITTI-13
    │   ├── KITTI-17
    │   ├── PETS09-S2L1
    │   ├── TUD-Campus
    │   ├── TUD-Stadtmitte
    │   ├── Venice-2
    │   └── Venice-6
    └── train
        ├── ADL-Rundle-1
        ├── ADL-Rundle-3
        ├── AVSS-AB
        ├── AVSS-AV
        ├── ETH-Bahnhof
        ├── ETH-Pedcross2
        ├── ETH-Sunnyday
        ├── KITTI-13
        ├── KITTI-17
        ├── PETS09-S2L1
        ├── TUD-Campus
        ├── TUD-Stadtmitte
        ├── Venice-2
└── yolo_dets
    ├── train
        ├── ADL-Rundle-1
        ├── ADL-Rundle-3
        ├── AVSS-AB
        ├── AVSS-AV
        ├── ETH-Bahnhof
        ├── ETH-Pedcross2
        ├── ETH-Sunnyday
        ├── KITTI-13
        ├── KITTI-17
        ├── PETS09-S2L1
        ├── TUD-Campus
        ├── TUD-Stadtmitte
        ├── Venice-2

└── outputs
```

## Run tracking on MOT15 train split

To run the code, please run the following command:

```bash
python main.py
```

This will run the code on the whole MOT15 training examples. To run the code on a different sequence, please change the path read in the file.

## Re-running detections with YOLO

To re-run the detections with YOLO, please run the following command:

```bash
python yolo_detection.py
```

This will run the YOLO detections on the MOT15 training examples. To run the code on a different sequence, please change the path read in the file.

## Running the evaluation

- Clone the MOTChallengeEvalKit
- Copy the `data/outputs` folder content to the `data/trackers/mot-challenge/MOT15-train` folder in the MOTChallengeEvalKit under the (new) directory name `CustomTracker`
- Run the following command:

```bash
python scripts/run_mot_challenge.py --BENCHMARK MOT15 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL CustomTracker --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --DO_PREPROC False
```

## Bechmarks

Benmarks results are available in the `benchmark` folder.
