import json
import torch 
import torchvision
from torch.utils.data import Dataset

class PropagandaDataset(Dataset):
    def __init__(self,data_dir,class_list, text_processor = None,image_processor=None):
      super(Dataset, self).__init__()
      self.data = self.load_data(data_dir)
      self.image_processor = image_processor
      self.text_processor = text_processor
      self.labels = {k: v for v, k in enumerate(class_list)}
        
    def __len__(self):
        return len(self.data)
        
    def load_data(self,data_dir):
        with open(data_dir, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def __getitem__(self, idx):
        item = {}
        sample = self.data[idx]
        text = sample["text"]
        if self.text_processor:
            text = self.text_processor(text)
        item['text'] = text
         
        if "batch_2" in sample['id']:
            id = int(sample['id'].split("_batch_2")[0]) + 2000
        else:
            id = int(sample['id'])
        item['id'] = torch.tensor(id, dtype=torch.int)

        lbls = torch.zeros(len(self.labels))
        lbls[[self.labels[tgt] for tgt in sample["labels"]]] = 1
        item['labels'] = lbls
        
        img = sample["image"]
        if self.image_processor:
            img = self.text_processor(img)
        item['image'] = img
        
        return item