"""LSTM model for sequential trajectory analysis"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/models.log")

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])
        return torch.FloatTensor(self.sequences[idx])

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.fc(last_output)
        
        return output

class LSTMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_columns = None
        self.scaler_mean = None
        self.scaler_std = None
        
        logger.info(f"Using device: {self.device}")
    
    def create_sequences(self, df, sequence_length=50):
        """Create sequences from trajectory data"""
        logger.info(f"Creating sequences with length {sequence_length}...")
        
        # Select feature columns
        exclude_cols = ['MMSI', 'timestamp', 'lat', 'lon', 'anomaly',
                       'lat_diff', 'lon_diff', 'geometry']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        sequences = []
        labels = []
        
        # Group by vessel
        for mmsi, group in df.groupby('MMSI'):
            group = group.sort_values('timestamp')
            
            # Get features
            features = group[self.feature_columns].values
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create sequences
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i+sequence_length]
                sequences.append(seq)
                
                # Label is max anomaly in sequence
                if 'anomaly' in group.columns:
                    label = group['anomaly'].iloc[i:i+sequence_length].max()
                    labels.append(label)
        
        sequences = np.array(sequences)
        
        # Normalize
        self.scaler_mean = sequences.mean(axis=(0, 1))
        self.scaler_std = sequences.std(axis=(0, 1)) + 1e-8
        sequences = (sequences - self.scaler_mean) / self.scaler_std
        
        logger.info(f"Created {len(sequences)} sequences")
        
        if labels:
            labels = np.array(labels)
            logger.info(f"Anomaly distribution: {np.bincount(labels.astype(int))}")
            return sequences, labels
        
        return sequences, None
    
    def train(self, df, sequence_length=50):
        """Train LSTM model"""
        logger.info("=" * 50)
        logger.info("LSTM MODEL TRAINING")
        logger.info("=" * 50)
        
        # Create sequences
        sequences, labels = self.create_sequences(df, sequence_length)
        
        # Split data
        split_idx = int(len(sequences) * 0.8)
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        # Create datasets
        train_dataset = TrajectoryDataset(train_seq, train_labels)
        test_dataset = TrajectoryDataset(test_seq, test_labels)
        
        batch_size = self.config.get('models', 'lstm', 'batch_size', default=32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = sequences.shape[2]
        hidden_size = self.config.get('models', 'lstm', 'hidden_size', default=128)
        num_layers = self.config.get('models', 'lstm', 'num_layers', default=2)
        dropout = self.config.get('models', 'lstm', 'dropout', default=0.3)
        
        self.model = LSTMAnomalyDetector(input_size, hidden_size, num_layers, dropout)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        epochs = self.config.get('models', 'lstm', 'epochs', default=50)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_seq, batch_labels in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss = self.evaluate(test_loader, criterion)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("LSTM training complete")
        
        return test_loader
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_seq, batch_labels in data_loader:
                batch_seq = batch_seq.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_seq)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def predict(self, sequences):
        """Predict anomaly scores"""
        self.model.eval()
        
        # Normalize
        sequences = (sequences - self.scaler_mean) / self.scaler_std
        
        dataset = TrajectoryDataset(sequences)
        loader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        
        with torch.no_grad():
            for batch_seq in loader:
                if isinstance(batch_seq, tuple):
                    batch_seq = batch_seq[0]
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def save_model(self, output_dir):
        """Save trained model"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_columns': self.feature_columns,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std
        }, output_dir / "lstm_model.pth")
        
        logger.info(f"LSTM model saved to {output_dir}")
    
    def load_model(self, model_dir, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        """Load trained model"""
        model_dir = Path(model_dir)
        
        checkpoint = torch.load(model_dir / "lstm_model.pth", map_location=self.device)
        
        self.model = LSTMAnomalyDetector(input_size, hidden_size, num_layers, dropout)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.feature_columns = checkpoint['feature_columns']
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        
        logger.info(f"LSTM model loaded from {model_dir}")
