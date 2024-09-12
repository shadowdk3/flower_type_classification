import os
import dill
import sys

from src.exception import CustomException

import torch

class LabelMapping:
    class_to_label = {
        0: 'daisy',
        1: 'dandelion',
        2: 'rose',
        3: 'sunflower',
        4: 'tulip',
    }

    label_to_class = {v: k for k, v in class_to_label.items()}

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_checkpoint(file_path):
    try:
        checkpoint = torch.load(file_path)
        return checkpoint
    
    except Exception as e:
        raise CustomException(e, sys)