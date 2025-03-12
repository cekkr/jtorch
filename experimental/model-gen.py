import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
import time
import copy
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np

class DynamicModel(nn.Module):
    """
    Modello dinamico che può essere costruito da una descrizione JSON
    """
    def __init__(self, architecture_config: Dict[str, Any]):
        """
        Costruisce un modello da una configurazione
        
        Args:
            architecture_config: Dizionario con la configurazione dell'architettura
                Formato:
                {
                    "input_dim": int,
                    "output_dim": int,
                    "input_precision": str,
                    "output_precision": str,
                    "layers": [
                        {"type": "linear", "dim": 128},
                        {"type": "conv1d", "dim": 64, "kernel_size": 3, "stride": 1},
                        ...
                    ]
                }
        """
        super(DynamicModel, self).__init__()
        
        self.input_dim = architecture_config["input_dim"]
        self.output_dim = architecture_config["output_dim"]
        self.input_precision = architecture_config.get("input_precision", "float32")
        self.output_precision = architecture_config.get("output_precision", "float32")
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.normalization = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Mappa che definisce le precision PyTorch
        self.precision_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16,
            "int8": torch.int8,
            "int32": torch.int32
        }
        
        # Costruisci i layer in base alla configurazione
        previous_dim = self.input_dim
        for i, layer_config in enumerate(architecture_config["layers"]):
            layer_type = layer_config["type"]
            current_dim = layer_config["dim"]
            
            # Aggiungi il layer in base al tipo
            if layer_type == "linear":
                self.layers.append(nn.Linear(previous_dim, current_dim))
            elif layer_type == "conv1d":
                kernel_size = layer_config.get("kernel_size", 3)
                stride = layer_config.get("stride", 1)
                padding = layer_config.get("padding", kernel_size // 2)
                self.layers.append(nn.Conv1d(
                    previous_dim, current_dim, kernel_size=kernel_size, 
                    stride=stride, padding=padding
                ))
            elif layer_type == "conv2d":
                kernel_size = layer_config.get("kernel_size", 3)
                stride = layer_config.get("stride", 1)
                padding = layer_config.get("padding", kernel_size // 2)
                self.layers.append(nn.Conv2d(
                    previous_dim, current_dim, kernel_size=kernel_size, 
                    stride=stride, padding=padding
                ))
            elif layer_type == "lstm":
                bidirectional = layer_config.get("bidirectional", False)
                self.layers.append(nn.LSTM(
                    previous_dim, current_dim // (2 if bidirectional else 1), 
                    batch_first=True, bidirectional=bidirectional
                ))
            elif layer_type == "gru":
                bidirectional = layer_config.get("bidirectional", False)
                self.layers.append(nn.GRU(
                    previous_dim, current_dim // (2 if bidirectional else 1), 
                    batch_first=True, bidirectional=bidirectional
                ))
            elif layer_type == "transformer":
                nhead = layer_config.get("heads", 8)
                dim_feedforward = layer_config.get("ff_dim", current_dim * 4)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=previous_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True
                )
                
                self.layers.append(nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=layer_config.get("num_layers", 1)
                ))
                # La dimensione del transformer rimane invariata
                current_dim = previous_dim
            else:
                raise ValueError(f"Tipo di layer non supportato: {layer_type}")
            
            # Aggiungi attivazione
            activation = layer_config.get("activation", "relu")
            if activation == "relu":
                self.activations.append(nn.ReLU())
            elif activation == "leaky_relu":
                self.activations.append(nn.LeakyReLU(0.1))
            elif activation == "elu":
                self.activations.append(nn.ELU())
            elif activation == "gelu":
                self.activations.append(nn.GELU())
            elif activation == "sigmoid":
                self.activations.append(nn.Sigmoid())
            elif activation == "tanh":
                self.activations.append(nn.Tanh())
            elif activation is None or activation.lower() == "none":
                self.activations.append(nn.Identity())
            else:
                raise ValueError(f"Attivazione non supportata: {activation}")
            
            # Aggiungi normalizzazione
            normalization = layer_config.get("normalization", None)
            if normalization == "batch":
                if layer_type in ["conv1d", "conv2d"]:
                    if layer_type == "conv1d":
                        self.normalization.append(nn.BatchNorm1d(current_dim))
                    else:  # conv2d
                        self.normalization.append(nn.BatchNorm2d(current_dim))
                else:
                    self.normalization.append(nn.BatchNorm1d(current_dim))
            elif normalization == "layer":
                self.normalization.append(nn.LayerNorm(current_dim))
            elif normalization is None or normalization.lower() == "none":
                self.normalization.append(nn.Identity())
            else:
                raise ValueError(f"Normalizzazione non supportata: {normalization}")
            
            # Aggiungi dropout
            dropout_rate = layer_config.get("dropout", 0.0)
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Aggiorna la dimensione per il prossimo layer
            previous_dim = current_dim
        
        # Layer di output
        if previous_dim != self.output_dim:
            self.output_layer = nn.Linear(previous_dim, self.output_dim)
        else:
            self.output_layer = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass del modello
        
        Args:
            x: Input di dimensione batch_size x input_dim
            
        Returns:
            Output di dimensione batch_size x output_dim
        """
        # Converti all'input precision richiesta
        x = x.to(self.precision_map[self.input_precision])
        
        # Applica i layer
        for i, (layer, activation, normalization, dropout) in enumerate(
            zip(self.layers, self.activations, self.normalization, self.dropouts)
        ):
            # Applica il layer
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                # Per RNN, il risultato è (output, hidden_state)
                x, _ = layer(x)
            else:
                x = layer(x)
            
            # Applica attivazione, normalizzazione e dropout
            x = activation(x)
            
            # Per Conv1d e Conv2d, la normalizzazione si aspetta che i canali siano nella seconda dimensione
            if isinstance(normalization, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Skip la normalizzazione se le dimensioni non corrispondono
                # Questo può accadere quando il batch è di dimensione 1
                if x.dim() > 1 and x.size(0) > 1:
                    x = normalization(x)
            else:
                x = normalization(x)
            
            x = dropout(x)
        
        # Applica il layer di output
        x = self.output_layer(x)
        
        # Converti all'output precision richiesta
        x = x.to(self.precision_map[self.output_precision])
        
        return x

class ModelManager:
    """
    Classe per gestire modelli PyTorch da specifiche JSON
    """
    def __init__(self, 
                 model_name: str = "dynamic_model",
                 save_dir: str = "./model_checkpoints",
                 log_dir: str = "./logs"):
        """
        Inizializza il manager
        
        Args:
            model_name: Nome del modello
            save_dir: Directory dove salvare i checkpoint
            log_dir: Directory dove salvare i log
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Crea le directory se non esistono
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Configura il logger
        self.setup_logger()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_model_state = None
        self.best_metric = float('-inf')
        self.start_epoch = 0
        self.config = None
    
    def setup_logger(self):
        """
        Configura il logger
        """
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)
        
        # Configura l'handler del file
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.model_name}.log"))
        file_handler.setLevel(logging.INFO)
        
        # Configura l'handler della console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Crea un formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Aggiungi gli handler al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def create_model_from_json(self, json_path: str) -> DynamicModel:
        """
        Crea un modello da un file JSON
        
        Args:
            json_path: Percorso del file JSON
            
        Returns:
            Il modello creato
        """
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        return self.create_model_from_config(config)
    
    def create_model_from_config(self, config: Dict[str, Any]) -> DynamicModel:
        """
        Crea un modello da una configurazione
        
        Args:
            config: Dizionario con la configurazione
            
        Returns:
            Il modello creato
        """
        self.config = config
        
        # Crea il modello
        self.model = DynamicModel(config["architecture"])
        
        # Configura l'ottimizzatore
        optimizer_config = config.get("optimizer", {"name": "adam", "lr": 0.001})
        self.configure_optimizer(optimizer_config)
        
        # Configura lo scheduler
        scheduler_config = config.get("scheduler", {"name": "none"})
        self.configure_scheduler(scheduler_config)
        
        # Configura la loss
        loss_config = config.get("loss", {"name": "cross_entropy"})
        self.configure_loss(loss_config)
        
        self.logger.info(f"Modello creato: {self.model}")
        self.logger.info(f"Totale parametri: {sum(p.numel() for p in self.model.parameters())}")
        
        return self.model
    
    def configure_optimizer(self, optimizer_config: Dict[str, Any]):
        """
        Configura l'ottimizzatore
        
        Args:
            optimizer_config: Configurazione dell'ottimizzatore
        """
        optimizer_name = optimizer_config["name"].lower()
        lr = optimizer_config.get("lr", 0.001)
        weight_decay = optimizer_config.get("weight_decay", 0)
        
        if optimizer_name == "adam":
            betas = optimizer_config.get("betas", (0.9, 0.999))
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
            )
        elif optimizer_name == "adamw":
            betas = optimizer_config.get("betas", (0.9, 0.999))
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
            )
        elif optimizer_name == "sgd":
            momentum = optimizer_config.get("momentum", 0.9)
            nesterov = optimizer_config.get("nesterov", False)
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay, nesterov=nesterov
            )
        elif optimizer_name == "rmsprop":
            alpha = optimizer_config.get("alpha", 0.99)
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Ottimizzatore non supportato: {optimizer_name}")
        
        self.logger.info(f"Ottimizzatore configurato: {self.optimizer}")
    
    def configure_scheduler(self, scheduler_config: Dict[str, Any]):
        """
        Configura lo scheduler
        
        Args:
            scheduler_config: Configurazione dello scheduler
        """
        scheduler_name = scheduler_config["name"].lower()
        
        if scheduler_name == "none" or scheduler_name is None:
            self.scheduler = None
        elif scheduler_name == "step":
            step_size = scheduler_config.get("step_size", 10)
            gamma = scheduler_config.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == "multistep":
            milestones = scheduler_config.get("milestones", [30, 60, 90])
            gamma = scheduler_config.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        elif scheduler_name == "plateau":
            patience = scheduler_config.get("patience", 10)
            factor = scheduler_config.get("factor", 0.1)
            mode = scheduler_config.get("mode", "min")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, patience=patience,
                verbose=True
            )
        elif scheduler_name == "cosine":
            T_max = scheduler_config.get("T_max", 100)
            eta_min = scheduler_config.get("eta_min", 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        else:
            raise ValueError(f"Scheduler non supportato: {scheduler_name}")
        
        self.logger.info(f"Scheduler configurato: {self.scheduler}")
    
    def configure_loss(self, loss_config: Dict[str, Any]):
        """
        Configura la loss
        
        Args:
            loss_config: Configurazione della loss
        """
        loss_name = loss_config["name"].lower()
        
        if loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == "bce":
            self.criterion = nn.BCELoss()
        elif loss_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        elif loss_name == "smooth_l1":
            beta = loss_config.get("beta", 1.0)
            self.criterion = nn.SmoothL1Loss(beta=beta)
        else:
            raise ValueError(f"Loss non supportata: {loss_name}")
        
        self.logger.info(f"Loss configurata: {self.criterion}")
    
    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """
        Salva un checkpoint del modello
        
        Args:
            epoch: Epoca corrente
            metric: Metrica di performance
            is_best: Se è il miglior modello finora
        """
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}.pt")
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint salvato: {checkpoint_path}")
        
        if is_best:
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Miglior modello salvato: {best_model_path}")
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_metric = metric
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        Carica un checkpoint del modello
        
        Args:
            checkpoint_path: Percorso del checkpoint
            load_optimizer: Se caricare anche l'ottimizzatore
            
        Returns:
            L'epoca del checkpoint
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint non trovato: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path)
        
        # Se il modello non è stato ancora creato, crealo dalla configurazione
        if self.model is None and 'config' in checkpoint:
            self.create_model_from_config(checkpoint['config'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_metric = checkpoint.get('metric', float('-inf'))
        epoch = checkpoint.get('epoch', 0)
        
        self.logger.info(f"Checkpoint caricato: {checkpoint_path}, Epoca: {epoch}")
        
        return epoch
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              metric_name: str = "accuracy",
              higher_is_better: bool = True,
              checkpoint_freq: int = 5,
              early_stopping_patience: Optional[int] = None,
              device: str = "cuda",
              resume: bool = False,
              resume_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Addestra il modello
        
        Args:
            train_loader: DataLoader per i dati di training
            val_loader: DataLoader per i dati di validation (opzionale)
            num_epochs: Numero di epoche
            metric_name: Nome della metrica da monitorare
            higher_is_better: Se valori più alti della metrica sono migliori
            checkpoint_freq: Frequenza di salvataggio dei checkpoint
            early_stopping_patience: Numero di epoche senza miglioramenti prima di fermarsi
            device: Device su cui eseguire il training
            resume: Se riprendere il training da un checkpoint
            resume_path: Percorso del checkpoint da cui riprendere
            
        Returns:
            Storia del training (loss e metriche)
        """
        # Sposta il modello sul device
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Storia del training
        history = {
            "train_loss": [],
            "val_loss": [],
            f"train_{metric_name}": [],
            f"val_{metric_name}": []
        }
        
        # Ripresa del training
        if resume:
            if resume_path:
                self.start_epoch = self.load_checkpoint(resume_path) + 1
            else:
                # Cerca l'ultimo checkpoint
                checkpoint_files = [f for f in os.listdir(self.save_dir) 
                                   if f.startswith(self.model_name) and f.endswith('.pt')]
                if checkpoint_files:
                    # Trova l'ultimo checkpoint per epoca
                    latest_checkpoint = max(checkpoint_files, 
                                           key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = os.path.join(self.save_dir, latest_checkpoint)
                    self.start_epoch = self.load_checkpoint(checkpoint_path) + 1
        
        # Inizializza le variabili per l'early stopping
        best_metric = float('-inf') if higher_is_better else float('inf')
        no_improvement_epochs = 0
        
        self.logger.info(f"Inizio training: {num_epochs} epoche, device: {device}")
        self.logger.info(f"Partenza dall'epoca: {self.start_epoch}")
        
        # Loop di training
        for epoch in range(self.start_epoch, num_epochs):
            # Training
            train_loss, train_metric = self._train_epoch(train_loader, device, metric_name)
            
            # Validation
            val_loss, val_metric = 0.0, 0.0
            if val_loader:
                val_loss, val_metric = self._validate_epoch(val_loader, device, metric_name)
            
            # Aggiorna lo scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Per ReduceLROnPlateau, passiamo la metrica di validation
                    metric_for_scheduler = val_loss if val_loader else train_loss
                    self.scheduler.step(metric_for_scheduler)
                else:
                    self.scheduler.step()
            
            # Aggiorna la storia
            history["train_loss"].append(train_loss)
            history[f"train_{metric_name}"].append(train_metric)
            if val_loader:
                history["val_loss"].append(val_loss)
                history[f"val_{metric_name}"].append(val_metric)
            
            # Log dei risultati
            log_msg = (f"Epoca {epoch+1}/{num_epochs} - "
                     f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}")
            if val_loader:
                log_msg += f", Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}"
            self.logger.info(log_msg)
            
            # Salva checkpoint periodicamente
            if (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(epoch + 1, val_metric if val_loader else train_metric)
            
            # Monitora la metrica per salvare il miglior modello
            metric_to_monitor = val_metric if val_loader else train_metric
            is_best = False
            
            if higher_is_better and metric_to_monitor > best_metric:
                best_metric = metric_to_monitor
                is_best = True
                no_improvement_epochs = 0
            elif not higher_is_better and metric_to_monitor < best_metric:
                best_metric = metric_to_monitor
                is_best = True
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
            
            if is_best:
                self.save_checkpoint(epoch + 1, metric_to_monitor, is_best=True)
            
            # Early stopping
            if early_stopping_patience and no_improvement_epochs >= early_stopping_patience:
                self.logger.info(f"Early stopping dopo {no_improvement_epochs} epoche senza miglioramenti")
                break
        
        # Carica il miglior modello alla fine del training
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path, load_optimizer=False)
        
        self.logger.info(f"Training completato. Miglior {metric_name}: {best_metric:.4f}")
        
        return history
    
    def _train_epoch(self, 
                     train_loader: DataLoader, 
                     device: torch.device,
                     metric_name: str) -> Tuple[float, float]:
        """
        Addestra il modello per un'epoca
        
        Args:
            train_loader: DataLoader per i dati di training
            device: Device su cui eseguire il training
            metric_name: Nome della metrica da calcolare
            
        Returns:
            Tuple con loss media e metrica
        """
        self.model.train()
        
        total_loss = 0.0
        batch_metrics = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            self.optimizer.step()
            
            # Metriche
            total_loss += loss.item()
            batch_metric = self._calculate_metric(outputs, target, metric_name)
            batch_metrics.append(batch_metric)
            
            # Log del progresso per batch lunghi
            if batch_idx % 100 == 0 and batch_idx > 0:
                self.logger.info(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_metric = np.mean(batch_metrics)
        
        return avg_loss, avg_metric
    
    def _validate_epoch(self, 
                        val_loader: DataLoader, 
                        device: torch.device,
                        metric_name: str) -> Tuple[float, float]:
        """
        Valuta il modello su un set di validation
        
        Args:
            val_loader: DataLoader per i dati di validation
            device: Device su cui eseguire la validation
            metric_name: Nome della metrica da calcolare
            
        Returns:
            Tuple con loss media e metrica
        """
        self.model.eval()
        
        total_loss = 0.0
        batch_metrics = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Metriche
                total_loss += loss.item()
                batch_metric = self._calculate_metric(outputs, target, metric_name)
                batch_metrics.append(batch_metric)
        
        avg_loss = total_loss / len(val_loader)
        avg_metric = np.mean(batch_metrics)
        
        return avg_loss, avg_metric
    
    def _calculate_metric(self, 
                         outputs: torch.Tensor, 
                         targets: torch.Tensor,
                         metric_name: str) -> float:
        """
        Calcola una metrica
        
        Args:
            outputs: Output del modello
            targets: Target
            metric_name: Nome della metrica
            
        Returns:
            Valore della metrica
        """
        if metric_name == "accuracy":
            if outputs.size() == targets.size():
                # Per problemi di regressione o multi-label
                predictions = (outputs > 0.5).float()
                correct = (predictions == targets).float().sum()
                metric = correct / (targets.size(0) * targets.size(1))
            else:
                # Per problemi di classificazione
                _, predictions = torch.max(outputs, 1)
                correct = (predictions == targets).sum().item()
                metric = correct / targets.size(0)
        elif metric_name == "mae":
            metric = nn.L1Loss()(outputs, targets).item()
        elif metric_name == "mse":
            metric = nn.MSELoss()(outputs, targets).item()
        else:
            raise ValueError(f"Metrica non supportata: {metric_name}")
        
        return metric
    
    def inference(self, 
                  data: Union[torch.Tensor, np.ndarray, List], 
                  device: str = "cuda") -> torch.Tensor:
        """
        Esegue inferenza con il modello
        
        Args:
            data: Input per il modello
            device: Device su cui eseguire l'inferenza
            
        Returns:
            Predizione del modello
        """
        # Verifica che il modello sia stato caricato
        if self.model is None:
            raise ValueError("Il modello non è stato caricato")
        
        # Converti l'input in un tensore se necessario
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Aggiungi una dimensione batch se necessario
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        
        # Sposta il modello sul device
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        data = data.to(device)
        
        # Esegui l'inferenza
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        
        return output
    
    def save_model_to_json(self, json_path: str):
        """
        Salva la configurazione del modello in un file JSON
        
        Args:
            json_path: Percorso dove salvare il file JSON
        """
        if self.config is None:
            raise ValueError("Nessuna configurazione disponibile")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
        
        self.logger.info(f"Configurazione salvata in: {json_path}")
    
    def export_model(self, 
                    export_dir: str,
                    export_formats: List[str] = ["pt"],
                    input_shape: Optional[Tuple] = None):
        """
        Esporta il modello in vari formati
        
        Args:
            export_dir: Directory di esportazione
            export_formats: Lista di formati da esportare ('pt', 'onnx', 'torchscript')
            input_shape: Forma dell'input del modello (opzionale)
        """
        os.makedirs(export_dir, exist_ok=True)
        
        if "pt" in export_formats:
            # Esporta il modello PyTorch
            torch.save(self.model.state_dict(), os.path.join(export_dir, f"{self.model_name}.pt"))
            self.logger.info(f"Modello PyTorch esportato in: {export_dir}/{self.model_name}.pt")
        
        if "torchscript" in export_formats:
            # Crea un dummy input se necessario
            if input_shape is None:
                dummy_input = torch.randn(1, self.model.input_dim)
            else:
                dummy_input = torch.randn(*input_shape)
                
            # Esporta il modello TorchScript
            script_model = torch.jit.trace(self.model, dummy_input)
            script_model.save(os.path.join(export_dir, f"{self.model_name}.torchscript"))
            self.logger.info(f"Modello TorchScript esportato in: {export_dir}/{self.model_name}.torchscript")
        
        if "onnx" in export_formats:
            try:
                import onnx
                
                # Crea un dummy input se necessario
                if input_shape is None:
                    dummy_input = torch.randn(1, self.model.input_dim)
                else:
                    dummy_input = torch.randn(*input_shape)
                
                # Esporta il modello ONNX
                onnx_path = os.path.join(export_dir, f"{self.model_name}.onnx")
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                )
                
                # Verifica il modello
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                self.logger.info(f"Modello ONNX esportato in: {onnx_path}")
            except ImportError:
                self.logger.warning("Libreria ONNX non installata. Installala con 'pip install onnx'")

def create_model_from_json(json_config: Union[str, Dict]) -> Tuple[DynamicModel, Dict]:
    """
    Funzione di utilità per creare un modello da una configurazione JSON
    
    Args:
        json_config: Percorso del file JSON o dizionario con la configurazione
        
    Returns:
        Tuple con il modello creato e la configurazione
    """
    # Carica la configurazione
    if isinstance(json_config, str):
        with open(json_config, 'r') as f:
            config = json.load(f)
    else:
        config = json_config
    
    # Crea il modello
    model = DynamicModel(config["architecture"])
    
    return model, config

def example_json_config():
    """
    Restituisce un esempio di configurazione JSON per un modello
    
    Returns:
        Dizionario con la configurazione
    """
    return {
        "model_name": "image_classifier",
        "architecture": {
            "input_dim": 784,  # MNIST flatten
            "output_dim": 10,  # 10 classi
            "input_precision": "float32",
            "output_precision": "float32",
            "layers": [
                {
                    "type": "linear",
                    "dim": 512,
                    "activation": "relu",
                    "normalization": "batch",
                    "dropout": 0.2
                },
                {
                    "type": "linear",
                    "dim": 256,
                    "activation": "relu",
                    "normalization": "batch",
                    "dropout": 0.2
                },
                {
                    "type": "linear",
                    "dim": 128,
                    "activation": "relu",
                    "normalization": "batch",
                    "dropout": 0.2
                }
            ]
        },
        "optimizer": {
            "name": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "scheduler": {
            "name": "cosine",
            "T_max": 100,
            "eta_min": 1e-6
        },
        "loss": {
            "name": "cross_entropy"
        }
    }

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un esempio di dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, input_dim=784, output_dim=10):
            self.data = torch.randn(num_samples, input_dim)
            self.targets = torch.randint(0, output_dim, (num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    # Crea i dataset
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)
    
    # Crea i dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Crea un manager
    manager = ModelManager(model_name="test_model")
    
    # Crea un modello da una configurazione
    config = example_json_config()
    model = manager.create_model_from_config(config)
    
    # Addestra il modello
    history = manager.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        metric_name="accuracy",
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_freq=1
    )
    
    # Esegui inferenza
    test_input = torch.randn(1, 784)
    output = manager.inference(test_input)
    print(f"Output shape: {output.shape}")
    
    # Esporta il modello
    manager.export_model("./exports", export_formats=["pt", "torchscript"])
    
    # Salva la configurazione
    manager.save_model_to_json("./exports/model_config.json")