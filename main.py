import json
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Union, Tuple, Optional, Callable
import logging
import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelParameterResolver:
    """Resolves parameters from the JSON configuration, handling relative references."""
    
    def __init__(self, parameters_config: List[Dict[str, Any]]):
        self.parameters = {}
        self.parameters_config = parameters_config
        self._resolve_parameters()
    
    def _resolve_parameters(self):
        """Resolve all parameters including those with relative references."""
        # First pass: resolve all direct parameters
        for param in self.parameters_config:
            if "relative_to" not in param:
                self.parameters[param["name"]] = param.get("default")
        
        # Second pass: resolve relative parameters
        resolved_all = False
        while not resolved_all:
            resolved_all = True
            for param in self.parameters_config:
                if param["name"] in self.parameters:
                    continue
                
                if "relative_to" in param:
                    try:
                        value = self._resolve_relative(param["relative_to"])
                        self.parameters[param["name"]] = value
                    except KeyError:
                        resolved_all = False
        
        # Third pass: resolve string references
        for param_name, param_value in self.parameters.items():
            if isinstance(param_value, str) and param_value in self.parameters:
                self.parameters[param_name] = self.parameters[param_value]
    
    def _resolve_relative(self, relative_expr):
        """Resolve a relative parameter expression like ["param1", "*", 0.5]."""
        if not isinstance(relative_expr, list):
            if isinstance(relative_expr, str) and relative_expr in self.parameters:
                return self.parameters[relative_expr]
            return relative_expr
        
        # Base case: just a parameter reference
        if len(relative_expr) == 1:
            param_name = relative_expr[0]
            if param_name not in self.parameters:
                raise KeyError(f"Parameter {param_name} not resolved yet")
            return self.parameters[param_name]
        
        # Process operations: +, -, *, /
        result = self._resolve_relative(relative_expr[0])
        i = 1
        while i < len(relative_expr):
            op = relative_expr[i]
            if op not in ["+", "-", "*", "/"]:
                raise ValueError(f"Unsupported operation: {op}")
            
            right_val = self._resolve_relative(relative_expr[i+1])
            
            if op == "+":
                result += right_val
            elif op == "-":
                result -= right_val
            elif op == "*":
                result *= right_val
            elif op == "/":
                result /= right_val
            
            i += 2
        
        return result
    
    def get_parameter(self, name: str) -> Any:
        """Get the value of a parameter by name."""
        return self.parameters.get(name)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all resolved parameters."""
        return self.parameters


class LayerFactory:
    """Creates PyTorch layers based on configuration."""
    
    @staticmethod
    def create_layer(layer_config: List, param_resolver: ModelParameterResolver) -> nn.Module:
        """Create a PyTorch layer from configuration."""
        layer_type = layer_config[0]
        
        if layer_type == "linear":
            in_features = param_resolver.get_parameter(layer_config[1])
            if len(layer_config) > 2:
                out_features = param_resolver.get_parameter(layer_config[2])
            else:
                out_features = in_features
            return nn.Linear(in_features, out_features)
        
        elif layer_type == "conv1d":
            in_channels = param_resolver.get_parameter(layer_config[1])
            if isinstance(layer_config[2], list):
                out_channels = LayerFactory._resolve_expression(layer_config[2], param_resolver)
            else:
                out_channels = param_resolver.get_parameter(layer_config[2])
            kernel_size = layer_config[3] if len(layer_config) > 3 else 3
            return nn.Conv1d(in_channels, out_channels, kernel_size)
        
        elif layer_type == "conv2d":
            in_channels = param_resolver.get_parameter(layer_config[1])
            if isinstance(layer_config[2], list):
                out_channels = LayerFactory._resolve_expression(layer_config[2], param_resolver)
            else:
                out_channels = param_resolver.get_parameter(layer_config[2])
            kernel_size = layer_config[3] if len(layer_config) > 3 else 3
            return nn.Conv2d(in_channels, out_channels, kernel_size)
        
        elif layer_type == "batchnorm1d":
            num_features = param_resolver.get_parameter(layer_config[1])
            return nn.BatchNorm1d(num_features)
        
        elif layer_type == "batchnorm2d":
            num_features = param_resolver.get_parameter(layer_config[1])
            return nn.BatchNorm2d(num_features)
        
        elif layer_type == "dropout":
            p = layer_config[1] if len(layer_config) > 1 else 0.5
            return nn.Dropout(p)
        
        elif layer_type == "relu":
            return nn.ReLU()
        
        elif layer_type == "tanh":
            return nn.Tanh()
        
        elif layer_type == "sigmoid":
            return nn.Sigmoid()
        
        elif layer_type == "softmax":
            dim = layer_config[1] if len(layer_config) > 1 else -1
            return nn.Softmax(dim=dim)
        
        elif layer_type == "embedding":
            num_embeddings = param_resolver.get_parameter(layer_config[1])
            embedding_dim = param_resolver.get_parameter(layer_config[2])
            return nn.Embedding(num_embeddings, embedding_dim)
        
        elif layer_type == "lstm":
            input_size = param_resolver.get_parameter(layer_config[1])
            hidden_size = param_resolver.get_parameter(layer_config[2])
            num_layers = layer_config[3] if len(layer_config) > 3 else 1
            bidirectional = layer_config[4] if len(layer_config) > 4 else False
            return nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        elif layer_type == "gru":
            input_size = param_resolver.get_parameter(layer_config[1])
            hidden_size = param_resolver.get_parameter(layer_config[2])
            num_layers = layer_config[3] if len(layer_config) > 3 else 1
            bidirectional = layer_config[4] if len(layer_config) > 4 else False
            return nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        elif layer_type == "flatten":
            return nn.Flatten()
        
        elif layer_type == "maxpool1d":
            kernel_size = layer_config[1]
            stride = layer_config[2] if len(layer_config) > 2 else kernel_size
            return nn.MaxPool1d(kernel_size, stride)
        
        elif layer_type == "maxpool2d":
            kernel_size = layer_config[1]
            stride = layer_config[2] if len(layer_config) > 2 else kernel_size
            return nn.MaxPool2d(kernel_size, stride)
        
        elif layer_type == "avgpool1d":
            kernel_size = layer_config[1]
            stride = layer_config[2] if len(layer_config) > 2 else kernel_size
            return nn.AvgPool1d(kernel_size, stride)
        
        elif layer_type == "avgpool2d":
            kernel_size = layer_config[1]
            stride = layer_config[2] if len(layer_config) > 2 else kernel_size
            return nn.AvgPool2d(kernel_size, stride)
        
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    @staticmethod
    def _resolve_expression(expr: List, param_resolver: ModelParameterResolver) -> int:
        """Resolve expressions like ["sentence_embedding", "+", 64]."""
        if len(expr) != 3:
            raise ValueError(f"Invalid expression format: {expr}")
        
        left = param_resolver.get_parameter(expr[0]) if isinstance(expr[0], str) else expr[0]
        op = expr[1]
        right = param_resolver.get_parameter(expr[2]) if isinstance(expr[2], str) else expr[2]
        
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            return left / right
        else:
            raise ValueError(f"Unsupported operation: {op}")


class DynamicModel(nn.Module):
    """A dynamic PyTorch model created from JSON configuration."""
    
    def __init__(self, model_config: List[Dict], param_resolver: ModelParameterResolver):
        super(DynamicModel, self).__init__()
        self.model_config = model_config
        self.param_resolver = param_resolver
        self.submodels = nn.ModuleDict()
        
        for submodel_config in model_config:
            name = submodel_config["name"]
            layers = submodel_config["layers"]
            
            submodel = nn.Sequential()
            for i, layer_config in enumerate(layers):
                layer = LayerFactory.create_layer(layer_config, param_resolver)
                submodel.add_module(f"{name}_{i}", layer)
            
            self.submodels[name] = submodel
        
        # Add hooks for potential customizations
        self._custom_forward_hooks = {}
    
    def register_custom_forward(self, name: str, hook: Callable):
        """Register a custom forward hook for a specific submodel."""
        self._custom_forward_hooks[name] = hook
    
    def forward(self, inputs: Dict[str, torch.Tensor], inference_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass using the specified inputs and inference configuration."""
        outputs = {}
        
        # Copy inputs to outputs so they're available for subsequent layers
        outputs.update(inputs)
        
        # Process all submodels
        for name, submodel in self.submodels.items():
            if name in self._custom_forward_hooks:
                # Use custom forward hook if registered
                outputs[name] = self._custom_forward_hooks[name](submodel, inputs, outputs)
            else:
                # Standard forward pass if input is available
                if name in inputs:
                    outputs[name] = submodel(inputs[name])
        
        return outputs


class DataProcessor:
    """Processes input and output data according to type definitions."""
    
    def __init__(self, type_definitions: List[Dict[str, Any]]):
        self.type_definitions = {t["name"]: t for t in type_definitions} if type_definitions else {}
    
    def process_input(self, input_config: Dict[str, Any], raw_data: Any) -> Any:
        """Process input data according to its type definition."""
        input_type = input_config.get("type")
        if input_type not in self.type_definitions:
            return raw_data  # No special processing
        
        type_def = self.type_definitions[input_type]
        processor_type = type_def.get("processor", "identity")
        
        if processor_type == "text_tokenizer":
            # Example text tokenizer processing
            max_length = type_def.get("max_length", 512)
            return self._tokenize_text(raw_data, max_length)
        
        elif processor_type == "image_processor":
            # Example image processor
            size = type_def.get("size", (224, 224))
            return self._process_image(raw_data, size)
        
        elif processor_type == "vector_normalizer":
            # Example vector normalizer
            return self._normalize_vector(raw_data)
        
        return raw_data
    
    def process_output(self, output_config: Dict[str, Any], model_output: Any) -> Any:
        """Process model output according to its type definition."""
        output_type = output_config.get("type")
        if output_type not in self.type_definitions:
            return model_output  # No special processing
        
        type_def = self.type_definitions[output_type]
        processor_type = type_def.get("output_processor", "identity")
        
        if processor_type == "token_to_text":
            return self._tokens_to_text(model_output)
        
        elif processor_type == "vector_to_class":
            classes = type_def.get("classes", [])
            return self._vector_to_class(model_output, classes)
        
        return model_output
    
    def _tokenize_text(self, text: str, max_length: int) -> List[int]:
        """Example tokenizer implementation."""
        # This is a placeholder. In a real application, you might use a proper tokenizer
        tokens = text.lower().split()
        return tokens[:max_length]
    
    def _process_image(self, image: Any, size: Tuple[int, int]) -> torch.Tensor:
        """Example image processor implementation."""
        # Placeholder for image processing
        return torch.rand((3, size[0], size[1]))
    
    def _normalize_vector(self, vector: Any) -> torch.Tensor:
        """Example vector normalizer."""
        vec = torch.tensor(vector, dtype=torch.float32)
        return vec / vec.norm()
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Example detokenizer implementation."""
        # Placeholder
        return " ".join([str(t) for t in tokens])
    
    def _vector_to_class(self, vector: torch.Tensor, classes: List[str]) -> str:
        """Example class predictor."""
        idx = torch.argmax(vector).item()
        if idx < len(classes):
            return classes[idx]
        return "unknown"


class CustomDataset(Dataset):
    """Custom dataset for handling the JSON-defined model inputs and outputs."""
    
    def __init__(self, 
                 data: List[Dict[str, Any]], 
                 input_configs: List[Dict[str, Any]], 
                 output_configs: List[Dict[str, Any]], 
                 data_processor: DataProcessor):
        self.data = data
        self.input_configs = input_configs
        self.output_configs = output_configs
        self.data_processor = data_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process inputs
        inputs = {}
        for config in self.input_configs:
            name = config["name"]
            if name in item:
                inputs[name] = self.data_processor.process_input(config, item[name])
        
        # Process expected outputs
        outputs = {}
        for config in self.output_configs:
            name = config["name"]
            if name in item:
                outputs[name] = self.data_processor.process_output(config, item[name])
        
        return inputs, outputs


class ModelManager:
    """Main class for managing model operations."""
    
    def __init__(self, config_file: str):
        """Initialize with a JSON configuration file."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.meta = self.config.get("meta", {})
        self.model_name = self.meta.get("name", "unnamed_model")
        
        # Initialize parameter resolver
        self.param_resolver = ModelParameterResolver(self.config.get("parameters", []))
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.config.get("type_definition", []))
        
        # Build the model
        self.model = DynamicModel(self.config.get("model", []), self.param_resolver)
        
        # Set default checkpoint directory
        self.checkpoint_dir = os.path.join("checkpoints", self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, training_name: str, data: List[Dict[str, Any]], 
              epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 0.001, checkpoint_interval: int = 5,
              resume_from: str = None, validation_data: List[Dict[str, Any]] = None,
              validation_interval: int = 1):
        """Train the model using the specified training configuration.
        
        Args:
            training_name: Name of the training configuration to use
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for the optimizer
            checkpoint_interval: How often to save checkpoints (in epochs)
            resume_from: Path to checkpoint to resume from
            validation_data: Optional validation data
            validation_interval: How often to run validation (in epochs)
        """
        # Find the training configuration
        training_config = None
        for tc in self.config.get("trainings", []):
            if tc["name"] == training_name:
                training_config = tc
                break
        
        if training_config is None:
            raise ValueError(f"Training configuration '{training_name}' not found")
        
        input_configs = training_config.get("input", [])
        output_configs = training_config.get("output", [])
        
        # Prepare dataset and dataloader
        dataset = CustomDataset(data, input_configs, output_configs, self.data_processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation dataset if provided
        val_dataloader = None
        if validation_data:
            val_dataset = CustomDataset(validation_data, input_configs, output_configs, self.data_processor)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss functions
        optimizer_type = training_config.get("optimizer", "adam")
        optimizer = self._get_optimizer(optimizer_type, learning_rate)
        
        loss_configs = training_config.get("losses", [])
        loss_functions = self._get_loss_functions(loss_configs)
        
        # Initialize training metrics tracking
        start_epoch = 0
        training_history = {
            'training_name': training_name,
            'epochs': [],
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
        
        # Resume from checkpoint if specified
        if resume_from:
            training_info = self.load_checkpoint(resume_from)
            if training_info:
                if 'epoch' in training_info:
                    start_epoch = training_info.get('epoch', 0) + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
                
                if 'training_history' in training_info:
                    training_history = training_info['training_history']
                    logger.info(f"Loaded training history with {len(training_history['epochs'])} epochs")
        
        # Training loop
        for epoch in range(start_epoch, start_epoch + epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            epoch_metrics = {'epoch': epoch + 1}
            
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_inputs)
                
                # Calculate loss
                loss = 0
                batch_losses = {}
                for loss_config in loss_configs:
                    output_name = loss_config["output"]
                    if output_name in outputs and output_name in batch_targets:
                        loss_name = loss_config["function"]
                        weight = loss_config.get("weight", 1.0)
                        loss_fn = loss_functions[loss_name]
                        
                        batch_loss = loss_fn(outputs[output_name], batch_targets[output_name])
                        weighted_loss = weight * batch_loss
                        loss += weighted_loss
                        batch_losses[f"{output_name}_{loss_name}_loss"] = batch_loss.item()
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Calculate training metrics
            epoch_loss = running_loss / len(dataloader)
            epoch_metrics['loss'] = epoch_loss
            
            # Add batch-specific losses
            for loss_name, loss_value in batch_losses.items():
                epoch_metrics[loss_name] = loss_value
            
            logger.info(f"Epoch {epoch+1}/{start_epoch + epochs}, Loss: {epoch_loss:.4f}")
            
            # Validation phase
            if val_dataloader and (epoch + 1) % validation_interval == 0:
                self.model.eval()
                val_running_loss = 0.0
                
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        # Forward pass
                        val_outputs = self.model(val_inputs)
                        
                        # Calculate loss
                        val_loss = 0
                        for loss_config in loss_configs:
                            output_name = loss_config["output"]
                            if output_name in val_outputs and output_name in val_targets:
                                loss_name = loss_config["function"]
                                weight = loss_config.get("weight", 1.0)
                                loss_fn = loss_functions[loss_name]
                                
                                batch_loss = loss_fn(val_outputs[output_name], val_targets[output_name])
                                val_loss += weight * batch_loss
                        
                        val_running_loss += val_loss.item()
                
                val_epoch_loss = val_running_loss / len(val_dataloader)
                epoch_metrics['val_loss'] = val_epoch_loss
                logger.info(f"Validation Loss: {val_epoch_loss:.4f}")
            
            # Update training history
            training_history['epochs'].append(epoch_metrics)
            
            # Save checkpoint if needed
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_info = {
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'training_name': training_name,
                    'training_config': training_config,
                    'training_history': training_history,
                    'optimizer_state': optimizer.state_dict()
                }
                
                if 'val_loss' in epoch_metrics:
                    checkpoint_info['val_loss'] = epoch_metrics['val_loss']
                
                self.save_checkpoint(
                    f"{training_name}_epoch_{epoch+1}", 
                    training_info=checkpoint_info
                )
        
        # Save final model
        final_info = {
            'epoch': start_epoch + epochs,
            'loss': epoch_loss,
            'training_name': training_name,
            'training_config': training_config,
            'training_history': training_history,
            'optimizer_state': optimizer.state_dict(),
            'is_final_checkpoint': True
        }
        
        if 'val_loss' in epoch_metrics:
            final_info['val_loss'] = epoch_metrics['val_loss']
            
        self.save_checkpoint(f"{training_name}_final", training_info=final_info)
        logger.info(f"Training completed: {training_name}")
    
    def inference(self, inference_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference using the specified inference configuration."""
        # Find the inference configuration
        inference_config = None
        for ic in self.config.get("inferences", []):
            if ic["name"] == inference_name:
                inference_config = ic
                break
        
        if inference_config is None:
            raise ValueError(f"Inference configuration '{inference_name}' not found")
        
        input_configs = inference_config.get("input", [])
        output_configs = inference_config.get("output", [])
        
        # Process inputs
        processed_inputs = {}
        for config in input_configs:
            name = config["name"]
            if name in inputs:
                processed_inputs[name] = self.data_processor.process_input(config, inputs[name])
        
        # Convert inputs to tensors
        tensor_inputs = {k: torch.tensor(v) for k, v in processed_inputs.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(tensor_inputs, inference_name)
        
        # Process outputs
        results = {}
        for config in output_configs:
            name = config["name"]
            if name in outputs:
                results[name] = self.data_processor.process_output(config, outputs[name])
        
        return results
    
    def save_checkpoint(self, checkpoint_name: str, training_info: Dict[str, Any] = None):
        """Save a model checkpoint with training information.
        
        Args:
            checkpoint_name: Name for the checkpoint
            training_info: Dictionary containing training metadata like epoch, loss, etc.
        
        Returns:
            Tuple containing paths to the model checkpoint and info JSON
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{checkpoint_name}_{timestamp}"
        model_filename = f"{base_filename}.pt"
        info_filename = f"{base_filename}.json"
        
        model_path = os.path.join(self.checkpoint_dir, model_filename)
        info_path = os.path.join(self.checkpoint_dir, info_filename)
        
        # Save model state and configuration
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'parameters': self.param_resolver.get_all_parameters()
        }
        
        torch.save(checkpoint, model_path)
        
        # Save additional training information
        if training_info is None:
            training_info = {}
            
        # Add metadata
        training_info.update({
            'checkpoint_name': checkpoint_name,
            'model_name': self.model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'parameters': self.param_resolver.get_all_parameters()
        })
        
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
            
        logger.info(f"Checkpoint saved: {model_path}")
        logger.info(f"Training info saved: {info_path}")
        
        return model_path, info_path
    
    def load_checkpoint(self, checkpoint_path: str, load_training_info: bool = True):
        """Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            load_training_info: Whether to load the associated training info JSON
            
        Returns:
            Optional dictionary with training info if load_training_info is True
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Check if the config matches
        checkpoint_config = checkpoint.get('config', {})
        if checkpoint_config.get('meta', {}).get('name') != self.model_name:
            logger.warning(f"Loading checkpoint for a different model: "
                          f"{checkpoint_config.get('meta', {}).get('name')} vs {self.model_name}")
        
        # Update the model with checkpoint state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        # Try to load the associated training info
        training_info = None
        if load_training_info:
            info_path = os.path.splitext(checkpoint_path)[0] + '.json'
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        training_info = json.load(f)
                    logger.info(f"Training info loaded: {info_path}")
                except Exception as e:
                    logger.error(f"Error loading training info: {e}")
            else:
                logger.warning(f"No training info found at {info_path}")
                
        return training_info
    
    def list_checkpoints(self, include_training_info: bool = True):
        """List all available checkpoints for this model.
        
        Args:
            include_training_info: Whether to include training info in the results
            
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoints = []
        
        # Get all checkpoint files
        model_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        
        for filename in model_files:
            path = os.path.join(self.checkpoint_dir, filename)
            base_name = os.path.splitext(filename)[0]
            info_path = os.path.join(self.checkpoint_dir, f"{base_name}.json")
            
            checkpoint_info = {
                'filename': filename,
                'path': path,
                'created': datetime.datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M:%S"),
                'has_info': os.path.exists(info_path)
            }
            
            # Load training info if requested and available
            if include_training_info and os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        training_info = json.load(f)
                    
                    # Add key training info fields
                    if 'epoch' in training_info:
                        checkpoint_info['epoch'] = training_info['epoch']
                    if 'loss' in training_info:
                        checkpoint_info['loss'] = training_info['loss']
                    if 'accuracy' in training_info:
                        checkpoint_info['accuracy'] = training_info['accuracy']
                    if 'checkpoint_name' in training_info:
                        checkpoint_info['checkpoint_name'] = training_info['checkpoint_name']
                    
                    # Store full training info
                    checkpoint_info['training_info'] = training_info
                except Exception as e:
                    logger.error(f"Error loading training info for {filename}: {e}")
            
            checkpoints.append(checkpoint_info)
        
        return sorted(checkpoints, key=lambda x: x['created'], reverse=True)
    
    def _get_optimizer(self, optimizer_config, learning_rate: float) -> torch.optim.Optimizer:
        """Get the optimizer based on the configuration.
        
        Args:
            optimizer_config: String (for backward compatibility) or dict with detailed config
            learning_rate: Default learning rate if not specified in config
            
        Returns:
            PyTorch optimizer instance
        """
        # Handle string config (backward compatibility)
        if isinstance(optimizer_config, str):
            optimizer_type = optimizer_config.lower()
            optimizer_params = {"lr": learning_rate}
        else:
            # Handle dictionary config
            optimizer_type = optimizer_config.get("type", "adam").lower()
            optimizer_params = optimizer_config.get("params", {})
            
            # Ensure learning rate is set
            if "lr" not in optimizer_params:
                optimizer_params["lr"] = learning_rate
        
        # Create the appropriate optimizer
        if optimizer_type == "adam":
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "adamw":
            return optim.AdamW(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "sgd":
            # Handle momentum if not specified
            if "momentum" not in optimizer_params:
                optimizer_params["momentum"] = 0.9
            return optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "adagrad":
            return optim.Adagrad(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "adadelta":
            return optim.Adadelta(self.model.parameters(), **optimizer_params)
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_type}, using Adam instead")
            return optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def _get_loss_functions(self, loss_configs: List[Dict[str, Any]]) -> Dict[str, Callable]:
        """Get loss functions based on configuration."""
        loss_functions = {}
        
        for config in loss_configs:
            loss_name = config["function"]
            
            if loss_name == "mse":
                loss_functions[loss_name] = nn.MSELoss()
            elif loss_name == "cross_entropy":
                loss_functions[loss_name] = nn.CrossEntropyLoss()
            elif loss_name == "bce":
                loss_functions[loss_name] = nn.BCELoss()
            elif loss_name == "bce_with_logits":
                loss_functions[loss_name] = nn.BCEWithLogitsLoss()
            elif loss_name == "l1":
                loss_functions[loss_name] = nn.L1Loss()
            elif loss_name == "smooth_l1":
                loss_functions[loss_name] = nn.SmoothL1Loss()
            else:
                logger.warning(f"Unsupported loss function: {loss_name}, using MSE instead")
                loss_functions[loss_name] = nn.MSELoss()
        
        return loss_functions


# Additional convenience methods
class CheckpointManager:
    """Helper class for managing checkpoints more effectively."""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    def find_latest_checkpoint(self, training_name=None):
        """Find the latest checkpoint for a specific training or any training."""
        checkpoints = self.model_manager.list_checkpoints()
        
        if not checkpoints:
            return None
            
        if training_name:
            # Filter for the specific training
            filtered = [c for c in checkpoints if training_name in c['filename']]
            if not filtered:
                return None
            return filtered[0]
        else:
            # Return the most recent
            return checkpoints[0]
    
    def find_best_checkpoint(self, metric='val_loss', mode='min'):
        """Find the best checkpoint according to a validation metric.
        
        Args:
            metric: Metric to compare ('val_loss', 'accuracy', etc.)
            mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
            
        Returns:
            Best checkpoint info or None if no checkpoints with metrics are found
        """
        checkpoints = self.model_manager.list_checkpoints()
        
        # Filter checkpoints that have the metric
        valid_checkpoints = [c for c in checkpoints if 
                            c.get('has_info') and 
                            metric in c.get('training_info', {})]
        
        if not valid_checkpoints:
            return None
            
        # Sort by the metric
        if mode == 'min':
            return sorted(valid_checkpoints, key=lambda x: x['training_info'][metric])[0]
        else:
            return sorted(valid_checkpoints, key=lambda x: x['training_info'][metric], reverse=True)[0]
    
    def resume_training(self, training_name, data, epochs=10, **kwargs):
        """Resume training from the latest checkpoint."""
        latest = self.find_latest_checkpoint(training_name)
        
        if latest:
            logger.info(f"Resuming training from checkpoint: {latest['path']}")
            return self.model_manager.train(
                training_name, 
                data, 
                epochs=epochs, 
                resume_from=latest['path'],
                **kwargs
            )
        else:
            logger.info(f"No checkpoint found for {training_name}, starting fresh")
            return self.model_manager.train(training_name, data, epochs=epochs, **kwargs)
    
    def auto_save_best(self, metric='val_loss', mode='min'):
        """Configure a hook to automatically save when a metric improves."""
        # This would be implemented with a custom callback during training
        pass


# Example usage
if __name__ == "__main__":
    # Example JSON path
    config_file = "model_config.json"
    
    # Initialize manager
    manager = ModelManager(config_file)
    checkpoint_manager = CheckpointManager(manager)
    
    # Example training data
    training_data = [
        {"sentence": "This is an example sentence", "tokens": [1, 2, 3, 4]},
        {"sentence": "Another example", "tokens": [5, 6]}
    ]
    
    # Optional validation data
    validation_data = [
        {"sentence": "Validation example", "tokens": [7, 8, 9]},
        {"sentence": "More validation", "tokens": [10, 11]}
    ]
    
    # Train the model with validation
    manager.train(
        "train_sentence_embedding", 
        training_data, 
        epochs=5,
        validation_data=validation_data,
        validation_interval=1
    )
    
    # Save a checkpoint with training info
    checkpoint_path, info_path = manager.save_checkpoint(
        "manual_save",
        training_info={"custom_metric": 0.95}
    )
    
    # Load a checkpoint and its info
    training_info = manager.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint info: {training_info}")
    
    # Resume training from latest checkpoint
    latest_checkpoint = checkpoint_manager.find_latest_checkpoint("train_sentence_embedding")
    if latest_checkpoint:
        manager.train(
            "train_sentence_embedding", 
            training_data, 
            resume_from=latest_checkpoint['path'],
            epochs=5
        )
    
    # Find best checkpoint by validation loss
    best_checkpoint = checkpoint_manager.find_best_checkpoint(metric='val_loss', mode='min')
    if best_checkpoint:
        manager.load_checkpoint(best_checkpoint['path'])
        print(f"Loaded best checkpoint with val_loss: {best_checkpoint['training_info']['val_loss']}")
    
    # Run inference
    inference_input = {"sentence": "Test inference"}
    result = manager.inference("infer_sentence_embedding", inference_input)
    print(result)
    
    # List checkpoints with detailed info
    checkpoints = manager.list_checkpoints(include_training_info=True)
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.get('checkpoint_name', checkpoint['filename'])
        created = checkpoint['created']
        
        # Show metrics if available
        metrics_str = ""
        if 'loss' in checkpoint:
            metrics_str += f", Loss: {checkpoint['loss']:.4f}"
        if 'val_loss' in checkpoint:
            metrics_str += f", Val Loss: {checkpoint['val_loss']:.4f}"
        if 'accuracy' in checkpoint:
            metrics_str += f", Accuracy: {checkpoint['accuracy']:.4f}"
        
        print(f"{checkpoint_name} - {created}{metrics_str}")