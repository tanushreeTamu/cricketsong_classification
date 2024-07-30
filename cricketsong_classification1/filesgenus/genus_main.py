import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import json
import librosa
from dataset import CustomDataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

torch.manual_seed(42)


class CricketClassifier(LightningModule):

  def __init__(self,label2id, id2label, num_classes, train_loader_len, test_loader_len):
    super().__init__()
    self.model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)
    self.criterion = nn.CrossEntropyLoss()
    self.training_step_outputs = []
    self.val_step_outputs = []
    self.train_dataloader_len = train_loader_len
    self.test_dataloader_len = test_loader_len

  def forward(self, x):
    return self.model(x)
      
        
  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    outputs = self(inputs)

    logits = outputs.logits
    loss = self.criterion(logits, labels)
    _, predicted = torch.max(logits, dim=1)
    metrics = {"train_loss_step": loss.item()}
    
    self.logger.log_metrics(metrics, step=self.global_step)
    train_outputs = {"loss": loss, "labels": labels, "predictions": predicted}
    self.training_step_outputs.append(train_outputs)
    #print(f"Train Loss (step): {loss:.4f}")
    return train_outputs

  def on_train_epoch_end(self):
    outputs = self.training_step_outputs
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    metrics = {"train_loss_epoch": avg_loss.item()}
    self.logger.log_metrics(metrics, step = self.global_step)
    
    all_labels = torch.cat([x['labels'] for x in outputs], dim = 0)
    all_predictions = torch.cat([x['predictions'] for x in outputs], dim = 0)

    #calculate accuracy
    accuracy = (all_predictions == all_labels).sum().item() / all_labels.size(0)

    #calculate precision, recall, f1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels.cpu(), all_predictions.cpu(), average='macro')

    # Log metrics using logger.log_metrics method
    metrics = {
        "train/accuracy": accuracy,
        "train/precision": precision,
        "train/recall": recall,
        "train/f1_score": f1_score,
    }
    self.logger.log_metrics(metrics, step=self.global_step)

    print(f"Training Loss (epoch): {avg_loss:.4f}")
    print("Training Metrics:", metrics)
    self.val_step_outputs.clear()


  def validation_step(self, batch, batch_idx):
    inputs, labels = batch
    outputs = self(inputs)
    logits = outputs.logits

    loss = self.criterion(logits, labels)
    _, predicted = torch.max(logits, dim=1)
    metrics = {"val_loss_step": loss.item()}
    self.logger.log_metrics(metrics, step=self.global_step)

    
    val_outputs = {"loss": loss,"labels": labels, "predictions": predicted}
    self.val_step_outputs.append(val_outputs)
    #print(f"Validation Loss (step): {loss:.4f}")

    # Return values to be used in validation_epoch_end
    return val_outputs

  def on_validation_epoch_end(self):
    print("On validation epoch end")
    outputs = self.val_step_outputs

    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    metrics = {"val_loss_epoch": avg_loss.item()}
    self.logger.log_metrics(metrics, step = self.global_step)


    all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
    all_predictions = torch.cat([x['predictions'] for x in outputs], dim=0)

    # Calculate accuracy
    accuracy = (all_predictions == all_labels).sum().item() / all_labels.size(0)
    
    # Calculate precision, recall, F1-score, and confusion matrix
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels.cpu(), all_predictions.cpu(), average='macro')

    # Log metrics using logger.log_metrics method
    metrics = {
        "val/accuracy": accuracy,
        "val/precision": precision,
        "val/recall": recall,
        "val/f1_score": f1_score,
    }
    self.logger.log_metrics(metrics, step=self.global_step)

    print(f"Validation Loss (epoch): {avg_loss:.4f}")
    print("Validation Metrics:", metrics)
    self.val_step_outputs.clear()

  def configure_optimizers(self):
    print("configure optimizer")
    optimizer = optim.Adam(self.parameters(), lr=3e-5)
    num_training_steps = self.train_dataloader_len * self.trainer.max_epochs
    warmup_steps = int(num_training_steps * 0.1)
    scheduler = {
        'scheduler': OneCycleLR(optimizer, max_lr=3e-5, total_steps=num_training_steps, anneal_strategy='linear', pct_start=warmup_steps/num_training_steps, div_factor=25.0, final_div_factor=10000.0),
        'interval': 'step',
        'frequency': 1,
    }
    return [optimizer], [scheduler]
        

def main(train_data_path,test_data_path,epochs):
  train_set = CustomDataset(datapath = train_data_path, type = 'genus')

  label2id = train_set.label2id
  id2label = train_set.id2label
  print("Printing label2id\n")
  pprint.pprint(label2id)
  num_classes = len(label2id)

  test_set = CustomDataset(datapath = test_data_path, label2id = label2id, id2label = id2label, type = 'genus')
  

  lr_monitor = LearningRateMonitor(logging_interval="step")

  # Set up TensorBoard logger
  tb_logger = TensorBoardLogger("log/3_genus/LR3e-5", name="BS4_ep10_trial1_imbalanced_LR3e-5_correct")
  tb_logger.log_hyperparams({"id2label" : json.dumps(id2label)})

  
  # put the checkpoint to save the model at each epoch  
  checkpoint_callback = ModelCheckpoint(
      dirpath = "/gpfs/proj1/choe_lab/tanu/Genus_classification/saved_model_genus/3_genus/LR3e-5/trial1_BS4_imbalanced_ep10_correct",
      filename = "{epoch}",
      save_top_k = -1,
      
  ) 
    
   
  num_workers = 8 # or another value based on your system's specifications
  train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=num_workers)
  test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=num_workers)
  
  # Initialize the LightningModule
  classifier = CricketClassifier(label2id, id2label, num_classes, len(train_loader), len(test_loader))


  # Initialize the Trainer
  trainer = Trainer(
      max_epochs=epochs,
      devices=2,
      accelerator="gpu",
      strategy="ddp",
      logger=tb_logger,
      gradient_clip_val=4.0,
      accumulate_grad_batches=4,
      callbacks=[lr_monitor, checkpoint_callback]
  )
  
  # Train the model
  trainer.fit(classifier, train_loader, test_loader)


if __name__ == '__main__':

  train_data_path = "/gpfs/proj1/choe_lab/tanu/Genus_classification/features_extracted/3_genus/train/cricket_data_feature_extracted.pt"
  test_data_path = "/gpfs/proj1/choe_lab/tanu/Genus_classification/features_extracted/3_genus/test/cricket_data_feature_extracted.pt"

  main(train_data_path,test_data_path,epochs = 10)