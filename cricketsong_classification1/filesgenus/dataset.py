import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, datapath, label2id = None, id2label = None, type = "genus"):

    self.data = torch.load(datapath)
    if type == "genus":
      for item in self.data:
        item['label'] = item['label'].split(' ')[0]
    elif type == "species":
      for item in self.data:
        item['label'] = item['label'].split(' ')[1]

    if label2id == None and id2label == None:
      labels = sorted(set(item['label'] for item in self.data))
      label2id = {label:idx for idx,label in enumerate(labels)}
      id2label = {idx: label for label, idx in label2id.items()}
      self.label2id = label2id
      self.id2label = id2label

    else:
      self.label2id = label2id
      self.id2label = id2label


  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
        waveform_array = self.data[idx]['array']
        label = self.data[idx]['label']
        label_id = self.label2id[label]
        return torch.tensor(waveform_array, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)