import os
import sys
from src.exception import CustomException
from src.logger import logging
import shutil

from dataclasses import dataclass

from torchvision import datasets
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train")
    test_data_path: str=os.path.join('artifacts', "test")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        self.class_to_label = {
            0: 'daisy',
            1: 'dandelion',
            2: 'rose',
            3: 'sunflower',
            4: 'tulip',
        }
                
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or compont")
        try:
            dataset = datasets.ImageFolder(root='data/raw')

            logging.info("Create Training Folder")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            logging.info("Train test split initiated")
            train_data, test_data, train_label, test_label = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)
            
            # clear pervious data
            self.clean_folder(self.ingestion_config.train_data_path)
            self.clean_folder(self.ingestion_config.test_data_path)
            
            self.save_data_to_file(self.ingestion_config.train_data_path, train_data)
            self.save_data_to_file(self.ingestion_config.test_data_path, test_data)
    
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
                
        except Exception as e:
            return CustomException(e, sys)

    def save_data_to_file(self, data_folder, data):
        try:
            cnt = 0
            for image_path, label in data:
                img_name = os.path.basename(image_path)
                target_folder = os.path.join(data_folder, self.class_to_label.get(label))
                os.makedirs(target_folder, exist_ok=True)
                target_path = os.path.join(target_folder, img_name)

                shutil.copy2(image_path, target_path)
                cnt += 1
            logging.info(f"Number of files {cnt} copy to {data_folder}")
            
        except Exception as e:
            return CustomException(e, sys)
        
    def clean_folder(self, data_folder):
        try:
            file_cnt = 0
            folder_cnt = 0
            for root, dirs, files in os.walk(data_folder):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.remove(file_path)
                    file_cnt += 1
                    
                if root != data_folder:
                    os.rmdir(root)
                    folder_cnt += 1
                    
            logging.info(f"Number of files {file_cnt} deleted in {data_folder}")
            logging.info(f"Number of folder {folder_cnt} deleted in {data_folder}")
                        
        except Exception as e:
            return CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_loader, test_loader, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    best_epoch, best_accuracy, best_loss = modeltrainer.initiate_model_trainer(train_loader, test_loader)
    
    logging.info(f'The best model is trained at {best_epoch} epoch: accuracy - {best_accuracy}, loss - {best_loss}')