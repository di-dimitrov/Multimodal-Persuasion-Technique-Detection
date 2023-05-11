import os

import pandas as pd
from PIL import Image
from torch.utils.data import *
from PIL import ImageFile
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TaskDataset(Dataset):
    """
    Dataset class for single-task image classification
    """

    def __init__(self, file_path, task_name, sep, root_dir, file_type="csv",transform=None):
        """

        :param file_path: File containing image file path and labels
        :param sep: separator to read label csv file
        :param root_dir: root directory for image files
        :param transform: PIL transforms to apply
        """
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        # df = pd.read_csv(file_path, sep=sep, dtype=str)
        # self.X = df['image_path'].tolist()
        # self.y = df[task_name].tolist()
        self.X=[]
        self.y=[]
        if (file_type == "csv"):
            df = pd.read_csv(file_path, sep=sep, na_filter=False)

            for index, row in df.iterrows():
                try:
                    img_path = row['image_path']
                    label = str(row[task_name])
                    self.X.append(img_path)
                    self.y.append(label)
                except KeyError as e:
                    print("Error in %s:" % (e))
        elif (file_type == "jsonl"):
            with open(file_path, 'r') as file:
                for line in file:
                    json_obj = json.loads(line)
                    img_path = json_obj['img']
                    label = str(json_obj[task_name])
                    self.X.append(img_path)
                    self.y.append(label)
        elif (file_type == "json"):
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            for json_obj in data:
                img_path = json_obj['image']
                label = str(json_obj[task_name])
                self.X.append(img_path)
                self.y.append(label)

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = list(zip(self.X, [self.class_to_idx[i] for i in self.y]))

        print("data size: {}".format(len(self.samples)))

    def __getitem__(self, index):
        path, label = self.samples[index]
        f = open(os.path.join(self.root_dir, path), 'rb')
        img = Image.open(f)
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _find_classes(self):
        classes_set = set(self.y)
        classes = list(classes_set)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
