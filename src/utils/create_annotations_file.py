import os
from ..config import DevConfig
import pandas as pd

actions = os.listdir(DevConfig.ROOT_DIR)

action_label_df = pd.DataFrame({"action": actions, "label": [i for i in range(len(actions))]})
action_label_df.to_csv(os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "actions_label.csv"), index=False)

annotations = {"video_name": [], "action": [], "action_label": []}

for index, row in action_label_df.iterrows():
    video_names = os.listdir(os.path.join(DevConfig.ROOT_DIR, row["action"]))
    for video_name in video_names:
        annotations["video_name"].append(video_name)
        annotations["action"].append(row["action"])
        annotations["action_label"].append(row["label"])

annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv(os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "annotations.csv"), index = False)