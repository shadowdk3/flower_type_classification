import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from src.utils import save_object

class_to_label = {
    0: 'daisy',
    1: 'dandelion',
    2: 'rose',
    3: 'sunflower',
    4: 'tulip',
}

label_to_class = {v: k for k, v in class_to_label.items()}

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")
    
class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        
        datafiles = []
        for root, dirs, files in os.walk(dataset):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                datafiles.append(file_path)
                
        self.dataset = self.checkChannel(datafiles) # some images are CMYK, Grayscale, check only RGB 
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        image = Image.open(self.dataset[item])
        
        path_parts = self.dataset[item].split('/')
        className = path_parts[-2]
        classCategory = label_to_class.get(className)
        
        if self.transform:
            image = self.transform(image)
        return image, classCategory
        
    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if (Image.open(dataset[index]).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            return transform
         
        except Exception as e:
            raise CustomException(e, sys)
        
    def data_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = models.resnet50(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False
                
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)

            model.to(device)
            
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_dataset = ImageLoader(train_path, transform=self.get_data_transformer_object())
            test_dataset = ImageLoader(test_path, transform=self.get_data_transformer_object())
               
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)         
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)         

            preprocessing_obj = self.data_model()
                   
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_loader,
                test_loader,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
                        
        except Exception as e:
            raise CustomException(e, sys)