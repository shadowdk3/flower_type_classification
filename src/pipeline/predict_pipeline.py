import sys

from src.exception import CustomException
from src.utils import load_object, load_checkpoint

from PIL import Image
from torch.utils.data import DataLoader
import torch
        
class PredicPipeline:
    def __init__(self):
        model_path = 'artifacts/model.pkl'
        checkpoint_path = 'artifacts/checkpoint.pkl'

        object = load_object(model_path)
        checkpoint = load_checkpoint(checkpoint_path)     
    
        self.model = object['model']
        self.class_to_label = object['class_to_label']
        self.optimizer = object['optimizer']
        self.data_transforms = object['transform']
        self.device = object['device']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
    def predict(self, filepath):
        try:
            img_array = Image.open(filepath).convert("RGB")

            img = self.data_transforms(img_array).unsqueeze(dim=0) # Returns a new tensor with a dimension of size one inserted at the specified position.
            load = DataLoader(img)

            pred_label = None
            
            for x in load:
                x=x.to(self.device)
                pred = self.model(x)
                _, preds = torch.max(pred, 1)
                class_ = preds[0].cpu().item()
                pred_label = self.class_to_label.get(class_)
                
            return pred_label
        
        except Exception as e:
            raise CustomException(e, sys)