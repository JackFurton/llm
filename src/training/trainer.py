import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        device=None,
        checkpoint_dir="checkpoints",
        use_wandb=False,
        project_name="custom-llm"
    ):
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model and data
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Optimizer and scheduler
        self.optimizer = optimizer if optimizer else optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.scheduler = scheduler
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding token
        
        # Tracking and checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # WandB integration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name)
            wandb.watch(self.model)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            # Calculate loss (reshape for cross entropy)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_dataloader)
        self.train_losses.append(avg_loss)
        
        # Log time taken
        elapsed = time.time() - start_time
        print(f"Epoch completed in {elapsed:.2f}s. Average training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self):
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_dataloader)
        self.val_losses.append(avg_loss)
        
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, num_epochs, save_every=1, eval_every=1):
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = None
            if self.val_dataloader and epoch % eval_every == 0:
                val_loss = self.evaluate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt")
                    print(f"New best validation loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Log to WandB
            if self.use_wandb:
                log_dict = {"train_loss": train_loss}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Plot training curve
        self.plot_training_curve()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {path}")
    
    def plot_training_curve(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curve.png'))
        plt.close()
        
        print(f"Training curve saved to {os.path.join(self.checkpoint_dir, 'training_curve.png')}")
