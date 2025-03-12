import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional

class ModelArchitectDataset(Dataset):
    """
    Dataset che contiene informazioni su modelli e le loro prestazioni.
    Ogni elemento è composto da:
    - embedding della descrizione degli input
    - dimensione dell'input
    - dimensione dell'output
    - tipo e dimensione dei layer ottimali (target)
    - precisione ottimale (target)
    """
    def __init__(self, 
                input_embeddings: List[torch.Tensor], 
                input_dims: List[int], 
                output_dims: List[int],
                optimal_architectures: List[List[Dict[str, int]]], 
                optimal_precisions: List[Dict[str, str]]):
        """
        Args:
            input_embeddings: Lista di tensori rappresentanti gli embedding delle descrizioni degli input
            input_dims: Lista di dimensioni degli input
            output_dims: Lista di dimensioni degli output
            optimal_architectures: Lista di liste di dizionari che specificano tipo e dimensione di ogni layer
                Es: [[{'linear': 128}, {'linear': 64}], ...]
            optimal_precisions: Lista di dizionari con precisione ottimale per input e output
                Es: [{'input': 'float32', 'output': 'float16'}, ...]
        """
        self.input_embeddings = input_embeddings
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.optimal_architectures = optimal_architectures
        self.optimal_precisions = optimal_precisions
        
        # Verifica che tutte le liste abbiano la stessa lunghezza
        assert len(input_embeddings) == len(input_dims) == len(output_dims) == len(optimal_architectures) == len(optimal_precisions)
        
        # Determina la dimensione massima dell'architettura per il padding
        self.max_arch_len = max(len(arch) for arch in optimal_architectures)
        
        # Mappa i tipi di layer a interi
        self.layer_types = {'linear': 0, 'conv1d': 1, 'conv2d': 2, 'lstm': 3, 'gru': 4, 'transformer': 5}
        
        # Mappa le precisioni a interi
        self.precision_types = {'float16': 0, 'float32': 1, 'float64': 2, 'int8': 3, 'int32': 4, 'bfloat16': 5}
    
    def __len__(self):
        return len(self.input_embeddings)
    
    def __getitem__(self, idx):
        # Prepara l'input
        embedding = self.input_embeddings[idx]
        input_dim = torch.tensor(self.input_dims[idx], dtype=torch.float32)
        output_dim = torch.tensor(self.output_dims[idx], dtype=torch.float32)
        
        # Prepara il target per l'architettura
        architecture = self.optimal_architectures[idx]
        arch_types = []
        arch_dims = []
        
        # Crea le sequenze per i tipi di layer e le loro dimensioni
        for layer in architecture:
            for layer_type, dim in layer.items():
                arch_types.append(self.layer_types[layer_type])
                arch_dims.append(dim)
        
        # Padding per architetture di lunghezza diversa
        while len(arch_types) < self.max_arch_len:
            arch_types.append(-1)  # -1 indica padding
            arch_dims.append(0)
        
        # Prepara il target per le precisioni
        precision = self.optimal_precisions[idx]
        input_precision = self.precision_types[precision['input']]
        output_precision = self.precision_types[precision['output']]
        
        # Converti in tensori
        arch_types = torch.tensor(arch_types, dtype=torch.long)
        arch_dims = torch.tensor(arch_dims, dtype=torch.float32)
        input_precision = torch.tensor(input_precision, dtype=torch.long)
        output_precision = torch.tensor(output_precision, dtype=torch.long)
        
        return {
            'embedding': embedding,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'arch_types': arch_types,
            'arch_dims': arch_dims,
            'input_precision': input_precision,
            'output_precision': output_precision
        }

class ModelArchitect(nn.Module):
    """
    Modello che predice l'architettura ottimale e la precisione dato un embedding del problema
    """
    def __init__(self, 
                embedding_dim: int, 
                hidden_dim: int = 256, 
                num_heads: int = 4,
                num_layers: int = 3,
                max_arch_len: int = 10,
                num_layer_types: int = 6,
                num_precision_types: int = 6):
        """
        Args:
            embedding_dim: Dimensione dell'embedding dell'input
            hidden_dim: Dimensione dello stato nascosto
            num_heads: Numero di teste di attenzione
            num_layers: Numero di layer transformer
            max_arch_len: Lunghezza massima dell'architettura da predire
            num_layer_types: Numero di tipi di layer possibili
            num_precision_types: Numero di tipi di precisione possibili
        """
        super(ModelArchitect, self).__init__()
        
        # Layer per processare le dimensioni di input e output
        self.dim_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Layer per processare l'embedding
        self.embedding_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combina le rappresentazioni
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer per modellare la sequenza di layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(max_arch_len, hidden_dim)
        
        # Layer per predire il tipo di ogni layer dell'architettura
        self.layer_type_predictor = nn.Linear(hidden_dim, num_layer_types)
        
        # Layer per predire la dimensione di ogni layer
        self.layer_dim_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Layer per predire la precisione di input e output
        self.precision_predictor = nn.Linear(hidden_dim, num_precision_types * 2)
        
        self.max_arch_len = max_arch_len
    
    def forward(self, embedding, input_dim, output_dim):
        batch_size = embedding.shape[0]
        
        # Codifica le dimensioni di input e output
        dims = torch.stack([input_dim, output_dim], dim=1)
        dim_features = self.dim_encoder(dims)
        
        # Codifica l'embedding
        emb_features = self.embedding_encoder(embedding)
        
        # Combina le rappresentazioni
        combined = torch.cat([emb_features, dim_features], dim=1)
        features = self.fusion_layer(combined)
        
        # Espandi per generare la sequenza di layer
        seq_features = features.unsqueeze(1).expand(-1, self.max_arch_len, -1)
        
        # Aggiungi positional encoding
        positions = torch.arange(0, self.max_arch_len, device=embedding.device).expand(batch_size, self.max_arch_len)
        pos_encoding = self.pos_encoder(positions)
        seq_features = seq_features + pos_encoding
        
        # Applica il transformer
        transformer_out = self.transformer(seq_features)
        
        # Predici il tipo di ogni layer
        layer_types_logits = self.layer_type_predictor(transformer_out)
        
        # Predici la dimensione di ogni layer
        layer_dims = self.layer_dim_predictor(transformer_out).squeeze(-1)
        
        # Predici la precisione
        precision_logits = self.precision_predictor(features)
        input_precision_logits, output_precision_logits = torch.chunk(precision_logits, 2, dim=1)
        
        return {
            'layer_types_logits': layer_types_logits,
            'layer_dims': layer_dims,
            'input_precision_logits': input_precision_logits,
            'output_precision_logits': output_precision_logits
        }

def train_model_architect(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """
    Addestra il ModelArchitect
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    layer_type_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignora il padding
    layer_dim_criterion = nn.MSELoss(reduction='none')
    precision_criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            embedding = batch['embedding'].to(device)
            input_dim = batch['input_dim'].to(device)
            output_dim = batch['output_dim'].to(device)
            arch_types = batch['arch_types'].to(device)
            arch_dims = batch['arch_dims'].to(device)
            input_precision = batch['input_precision'].to(device)
            output_precision = batch['output_precision'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(embedding, input_dim, output_dim)
            
            # Calcolo delle loss per i tipi di layer
            layer_type_loss = layer_type_criterion(
                outputs['layer_types_logits'].reshape(-1, outputs['layer_types_logits'].size(-1)), 
                arch_types.reshape(-1)
            )
            
            # Calcolo delle loss per le dimensioni dei layer (consideriamo solo i layer non padding)
            mask = (arch_types != -1).float()
            layer_dim_loss = layer_dim_criterion(outputs['layer_dims'], arch_dims)
            layer_dim_loss = (layer_dim_loss * mask).sum() / (mask.sum() + 1e-8)
            
            # Calcolo delle loss per le precisioni
            input_precision_loss = precision_criterion(outputs['input_precision_logits'], input_precision)
            output_precision_loss = precision_criterion(outputs['output_precision_logits'], output_precision)
            
            # Loss totale
            loss = layer_type_loss + layer_dim_loss + input_precision_loss + output_precision_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                embedding = batch['embedding'].to(device)
                input_dim = batch['input_dim'].to(device)
                output_dim = batch['output_dim'].to(device)
                arch_types = batch['arch_types'].to(device)
                arch_dims = batch['arch_dims'].to(device)
                input_precision = batch['input_precision'].to(device)
                output_precision = batch['output_precision'].to(device)
                
                outputs = model(embedding, input_dim, output_dim)
                
                layer_type_loss = layer_type_criterion(
                    outputs['layer_types_logits'].reshape(-1, outputs['layer_types_logits'].size(-1)), 
                    arch_types.reshape(-1)
                )
                
                mask = (arch_types != -1).float()
                layer_dim_loss = layer_dim_criterion(outputs['layer_dims'], arch_dims)
                layer_dim_loss = (layer_dim_loss * mask).sum() / (mask.sum() + 1e-8)
                
                input_precision_loss = precision_criterion(outputs['input_precision_logits'], input_precision)
                output_precision_loss = precision_criterion(outputs['output_precision_logits'], output_precision)
                
                loss = layer_type_loss + layer_dim_loss + input_precision_loss + output_precision_loss
                
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Aggiorna lo scheduler
            scheduler.step(val_loss)
            
            # Salva il miglior modello
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model_architect.pth')
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def predict_architecture(model, input_embedding, input_dim, output_dim, device='cuda'):
    """
    Predice l'architettura ottimale e la precisione dato un input
    
    Args:
        model: Il modello addestrato
        input_embedding: L'embedding della descrizione dell'input
        input_dim: La dimensione dell'input
        output_dim: La dimensione dell'output
        
    Returns:
        Un dizionario con l'architettura predetta e le precisioni
    """
    model.eval()
    
    # Converti in tensori e sposta sul device
    if not isinstance(input_embedding, torch.Tensor):
        input_embedding = torch.tensor(input_embedding, dtype=torch.float32)
    if not isinstance(input_dim, torch.Tensor):
        input_dim = torch.tensor(input_dim, dtype=torch.float32)
    if not isinstance(output_dim, torch.Tensor):
        output_dim = torch.tensor(output_dim, dtype=torch.float32)
    
    input_embedding = input_embedding.unsqueeze(0).to(device)
    input_dim = input_dim.unsqueeze(0).to(device)
    output_dim = output_dim.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_embedding, input_dim, output_dim)
        
        # Decodifica i tipi di layer
        layer_types_logits = outputs['layer_types_logits'][0]  # Prima dimensione del batch
        layer_types = torch.argmax(layer_types_logits, dim=1).cpu().numpy()
        
        # Decodifica le dimensioni dei layer
        layer_dims = outputs['layer_dims'][0].cpu().numpy()
        
        # Arrotonda le dimensioni dei layer al più vicino multiplo di 8 (per efficienza)
        layer_dims = np.round(layer_dims / 8) * 8
        layer_dims = np.clip(layer_dims, 8, None).astype(int)
        
        # Decodifica le precisioni
        input_precision_idx = torch.argmax(outputs['input_precision_logits'], dim=1)[0].item()
        output_precision_idx = torch.argmax(outputs['output_precision_logits'], dim=1)[0].item()
        
        # Mappa inversa dei tipi di layer e precisioni
        idx_to_layer = {v: k for k, v in model.layer_types.items()}
        idx_to_precision = {v: k for k, v in model.precision_types.items()}
        
        # Costruisci l'architettura
        architecture = []
        for i in range(model.max_arch_len):
            if layer_types[i] == -1:  # Padding
                break
            
            layer_type = idx_to_layer[layer_types[i]]
            dim = layer_dims[i]
            
            architecture.append({layer_type: dim})
        
        # Ottieni le precisioni
        input_precision = idx_to_precision[input_precision_idx]
        output_precision = idx_to_precision[output_precision_idx]
        
        return {
            'architecture': architecture,
            'input_precision': input_precision,
            'output_precision': output_precision
        }

# Esempio di utilizzo
if __name__ == "__main__":
    # Dati di esempio
    embedding_dim = 300  # Dimensione dell'embedding per la descrizione dell'input
    
    # Crea dati sintetici di esempio
    num_samples = 1000
    input_embeddings = [torch.randn(embedding_dim) for _ in range(num_samples)]
    input_dims = [np.random.randint(1, 1024) for _ in range(num_samples)]
    output_dims = [np.random.randint(1, 1024) for _ in range(num_samples)]
    
    layer_types = ['linear', 'conv1d', 'conv2d', 'lstm', 'gru', 'transformer']
    precision_types = ['float16', 'float32', 'float64', 'int8', 'int32', 'bfloat16']
    
    # Genera architetture optimali di esempio
    optimal_architectures = []
    for _ in range(num_samples):
        num_layers = np.random.randint(1, 5)
        architecture = []
        for _ in range(num_layers):
            layer_type = np.random.choice(layer_types)
            dim = np.random.randint(1, 17) * 64  # Multipli di 64 per semplicità
            architecture.append({layer_type: dim})
        optimal_architectures.append(architecture)
    
    # Genera precisioni ottimali di esempio
    optimal_precisions = []
    for _ in range(num_samples):
        input_prec = np.random.choice(precision_types)
        output_prec = np.random.choice(precision_types)
        optimal_precisions.append({'input': input_prec, 'output': output_prec})
    
    # Crea il dataset
    dataset = ModelArchitectDataset(
        input_embeddings=input_embeddings,
        input_dims=input_dims,
        output_dims=output_dims,
        optimal_architectures=optimal_architectures,
        optimal_precisions=optimal_precisions
    )
    
    # Dividi in train e validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Crea i dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Inizializza il modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelArchitect(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        max_arch_len=max(len(arch) for arch in optimal_architectures),
        num_layer_types=len(layer_types),
        num_precision_types=len(precision_types)
    ).to(device)
    
    # Imposta attributi richiesti per la predizione
    model.layer_types = dataset.layer_types
    model.precision_types = dataset.precision_types
    
    # Addestra il modello
    train_model_architect(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        device=device
    )
    
    # Carica il miglior modello
    model.load_state_dict(torch.load('best_model_architect.pth'))
    
    # Esempio di predizione
    test_embedding = torch.randn(embedding_dim)
    test_input_dim = 784  # es. MNIST
    test_output_dim = 10  # es. 10 classi
    
    prediction = predict_architecture(
        model=model,
        input_embedding=test_embedding,
        input_dim=test_input_dim,
        output_dim=test_output_dim,
        device=device
    )
    
    print("Architettura predetta:")
    for i, layer in enumerate(prediction['architecture']):
        print(f"Layer {i+1}: {layer}")
    print(f"Precisione input: {prediction['input_precision']}")
    print(f"Precisione output: {prediction['output_precision']}")