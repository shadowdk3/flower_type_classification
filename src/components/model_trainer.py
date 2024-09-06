import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

import torch
import torch.optim as optim 
import torch.nn as nn

from tqdm import tqdm

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')
    preprocessor_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    checkpoint_path = 'checkpoint'
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        os.makedirs(self.model_trainer_config.checkpoint_path, exist_ok=True)

        self.model = load_object(self.model_trainer_config.preprocessor_file_path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, device, model, data, targets):
        model.train()
        data = data.to(device=device)
        targets = torch.tensor(targets).to(device=device)  # Convert targets to a tensor
        scores = model(data)
        
        loss = self.criterion(scores, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        _, preds = torch.max(scores, 1)
        correct_predictions = torch.sum(preds == targets).item()
        total_samples = data.size(0)
        
        return loss, correct_predictions, total_samples
    
    def test(self, device, model, test_loader):
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for data, targets in test_loader:
                data = data.to(device=device)
                targets = targets.to(device=device)
                output = model(data)
                _, preds = torch.max(output, 1)
                correct_predictions += (preds == targets).sum().item()
                total_samples += data.size(0)
        
        return correct_predictions, total_samples
    
    def initiate_model_trainer(self, train_loader, test_loader):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logging.info("model load: %s", self.model)
            num_epoch = 10
            best_accuracy = 0
            best_epoch = 0
            best_loss = 0
            for epoch in range(0, num_epoch):
                correct_predictions  = 0.0
                total_samples  = 0
                train_correct_predictions  = 0.0
                train_total_samples  = 0
                train_accuracy = 0
                loop = tqdm(enumerate(train_loader), total=len(train_loader))
                losses = []
                
                for batch_idx, (data, targets) in loop:
                    loss, train_correct_predictions, train_total_samples = self.train(device, self.model, data, targets)
                    losses.append(loss)
                    train_correct_predictions += train_correct_predictions
                    train_total_samples += train_total_samples
                    
                    loop.set_description(f"Epoch {epoch+1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
                    loop.set_postfix(loss=loss.data.item())
                
                train_accuracy = train_correct_predictions / train_total_samples * 100
                logging.info(f'epoch {epoch}: loss={losses[epoch]}, accuracy={train_accuracy}')
                
                # Validation loop
                correct_predictions, total_samples = self.test(device, self.model, test_loader)

                # Calculate validation accuracy
                accuracy = correct_predictions / total_samples * 100

                # Save the best model based on validation accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_loss = loss
                    checkpoint_path = "./checkpoint/best_checkpoint.pt"
                    
                    # save model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'accuracy': accuracy
                    }, checkpoint_path)
            
            return best_epoch, best_accuracy, best_loss
        
        except Exception as e:
            raise CustomException(e, sys)
