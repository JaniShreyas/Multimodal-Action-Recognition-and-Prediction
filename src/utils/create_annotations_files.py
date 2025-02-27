import os
from ..config import DevConfig
import pandas as pd
from sklearn.model_selection import train_test_split

actions = os.listdir(DevConfig.ROOT_DIR)

action_label_df = pd.DataFrame(
    {"action": actions, "label": [i for i in range(len(actions))]}
)
action_label_df.to_csv(
    os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "actions_label.csv"), index=False
)

annotations = {"video_name": [], "action": [], "action_label": []}

for index, row in action_label_df.iterrows():
    video_names = os.listdir(os.path.join(DevConfig.ROOT_DIR, row["action"]))
    for video_name in video_names:
        annotations["video_name"].append(video_name)
        annotations["action"].append(row["action"])
        annotations["action_label"].append(row["label"])

annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv(
    os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "annotations.csv"), index=False
)

train_df, temp_df = train_test_split(
    annotations_df, test_size=0.3, random_state=DevConfig.RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=DevConfig.RANDOM_STATE
)

train_df.to_csv(
    os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "train_annotations.csv"), index=False
)
val_df.to_csv(
    os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "val_annotations.csv"), index=False
)
test_df.to_csv(
    os.path.join(DevConfig.ANNOTATIONS_DIR_LOCAL, "test_annotations.csv"), index=False
)
