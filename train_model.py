from roboflow import Roboflow
import os

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="7wEwFHtz2YN3aOiTTk8X")  # private key
project = rf.workspace("camouflage").project("soldier-civilian-detection")
dataset = project.version(2).download("yolov5")

# dataset.location contains the local directory where the dataset was downloaded
dataset_dir = dataset.location

# Build the training command using the dynamically obtained path
# command = (
#     f'python train.py --img 640 --batch 8 --epochs 50 '
#     f'--data "{dataset_dir}/data.yaml" --weights yolov5m.pt '
#     f'--cache --hyp hyp.scratch-high.yaml --project runs/train --name exp6 --exist-ok'
# )

command = (
    f'python train.py --img 768 --batch 8 --epochs 100 '
    f'--data "{dataset_dir}/data.yaml" --cache '
    f'--hyp hyp.scratch-high.yaml --project runs/train --name evolve_run --exist-ok --evolve'
)

# Run training
os.system(command)
 