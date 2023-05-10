import torch 
import torchvision
from torch.utils.data import Dataset

def load_data(data_dir):
    ...
    pass

class PropagandaDataset(Dataset):
    def __init__(self,data_dir,image_processor=None,text_processor,labels):
      super(Dataset, self).__init__()
      self.data = load_data(data_dir)
      self.image_processor = image_processor
      self.text_processor = text_processor
      self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        item.text = self.text_processor(data[idx]["text"])
      
        if "batch_2" in sample_info['id']:
            id = int(sample_info['id'].split("_batch_2")[0]) + 2000
        else:
            id = int(sample_info['id'])
        item.id = torch.tensor(id, dtype=torch.int)

        label = torch.zeros(22)
        label[[self.labels[tgt] for tgt in sample_info["labels"]]] = 1
        item.labels = label
        
        item.image = self.image_processor(self.data[idx]["image"])
        return item