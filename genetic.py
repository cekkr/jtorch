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
import tempfile

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
        """Train the model using the specified training configuration."""
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
        
        return training_history['epochs']
    
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
        """Save a model checkpoint with training information."""
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
        """Load a model from a checkpoint."""
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
        """List all available checkpoints for this model."""
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
        """Get the optimizer based on the configuration."""
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


class CheckpointManager:
    """Helper class for managing checkpoints more effectively."""
    
    def __init__(self, model_manager):
        """Initialize with a model manager."""
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
        """Find the best checkpoint according to a validation metric."""
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


class RandomSequenceGenerator:
    """Genera sequenze numeriche casuali o pseudocasuali da usare come genomi."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inizializza il generatore di sequenze casuali.
        
        Args:
            seed: Seme per la generazione pseudocasuale
        """
        self.rng = np.random.RandomState(seed)
    
    def uniform_sequence(self, length: int) -> List[float]:
        """
        Genera una sequenza con distribuzione uniforme tra 0 e 1.
        
        Args:
            length: Lunghezza della sequenza
            
        Returns:
            Lista di numeri casuali uniformi
        """
        return self.rng.random(length).tolist()
    
    def normal_sequence(self, length: int, mean: float = 0.5, std: float = 0.15) -> List[float]:
        """
        Genera una sequenza con distribuzione normale, troncata tra 0 e 1.
        
        Args:
            length: Lunghezza della sequenza
            mean: Media della distribuzione
            std: Deviazione standard
            
        Returns:
            Lista di numeri casuali con distribuzione normale
        """
        # Genera valori con distribuzione normale
        values = self.rng.normal(mean, std, length)
        
        # Tronca i valori tra 0 e 1
        values = np.clip(values, 0.0, 1.0)
        
        return values.tolist()
    
    def generate_from_hash(self, text: str, length: int) -> List[float]:
        """
        Genera una sequenza basata su un hash di testo.
        Utile per ottenere genomi riproducibili da stringhe.
        
        Args:
            text: Testo da cui generare l'hash
            length: Lunghezza della sequenza desiderata
            
        Returns:
            Lista di numeri derivati dall'hash
        """
        import hashlib
        
        # Genera un hash dal testo
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Usa l'hash come seme per il generatore
        seed = int(hash_hex, 16) % (2**32)
        local_rng = np.random.RandomState(seed)
        
        return local_rng.random(length).tolist()
    
    def chaotic_sequence(self, length: int, r: float = 3.9) -> List[float]:
        """
        Genera una sequenza usando la mappa logistica (sistema caotico).
        
        Args:
            length: Lunghezza della sequenza
            r: Parametro della mappa logistica (3.57 < r < 4 per comportamento caotico)
            
        Returns:
            Lista di numeri generati dalla mappa logistica
        """
        sequence = []
        
        # Valore iniziale casuale
        x = self.rng.random()
        
        # Genera la sequenza usando la mappa logistica: x_n+1 = r * x_n * (1 - x_n)
        for _ in range(length):
            x = r * x * (1 - x)
            sequence.append(x)
        
        return sequence
    
    def prime_based_sequence(self, length: int) -> List[float]:
        """
        Genera una sequenza basata sui primi numeri primi.
        
        Args:
            length: Lunghezza della sequenza
            
        Returns:
            Lista di numeri derivati dai numeri primi
        """
        sequence = []
        
        # Funzione semplice per verificare se un numero è primo
        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        # Trova i primi N numeri primi
        primes = []
        num = 2
        while len(primes) < length + 100:  # Genera più primi del necessario
            if is_prime(num):
                primes.append(num)
            num += 1
        
        # Usa i numeri primi per generare valori tra 0 e 1
        for i in range(length):
            # Prendi due numeri primi e usa il loro rapporto modulo 1
            p1 = primes[i]
            p2 = primes[i + 1]
            value = (p1 / p2) % 1
            sequence.append(value)
        
        return sequence
    
    def mixed_sequence(self, length: int) -> List[float]:
        """
        Genera una sequenza combinando diverse strategie.
        
        Args:
            length: Lunghezza della sequenza
            
        Returns:
            Lista di numeri generati con un mix di strategie
        """
        # Decide quanti numeri generare con ciascuna strategia
        segment_length = length // 3
        remaining = length - (segment_length * 3)
        
        # Genera segmenti con diverse strategie
        uniform_segment = self.uniform_sequence(segment_length)
        normal_segment = self.normal_sequence(segment_length)
        chaotic_segment = self.chaotic_sequence(segment_length)
        
        # Genera il segmento rimanente
        remaining_segment = self.prime_based_sequence(remaining) if remaining > 0 else []
        
        # Combina i segmenti
        combined = uniform_segment + normal_segment + chaotic_segment + remaining_segment
        
        # Mescola la sequenza
        self.rng.shuffle(combined)
        
        return combined



class GeneticModelGenerator:
    """
    Converte una sequenza numerica (genoma) in una configurazione di modello completa.
    Il genoma determina ogni aspetto del modello: struttura, parametri, optimizer, ecc.
    """
    
    # Definizione dei possibili tipi di layer
    LAYER_TYPES = [
        "linear", "conv1d", "conv2d", "batchnorm1d", "batchnorm2d",
        "dropout", "relu", "tanh", "sigmoid", "softmax", "embedding",
        "lstm", "gru", "flatten", "maxpool1d", "maxpool2d", "avgpool1d", "avgpool2d"
    ]
    
    # Definizione dei possibili tipi di ottimizzatori
    OPTIMIZER_TYPES = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta"]
    
    # Definizione dei possibili tipi di funzioni di loss
    LOSS_FUNCTIONS = ["mse", "cross_entropy", "bce", "bce_with_logits", "l1", "smooth_l1"]
    
    def __init__(self, 
                 seed: Optional[int] = None, 
                 complexity_range: Tuple[int, int] = (2, 10),
                 parameter_range: Tuple[int, int] = (8, 512),
                 model_base_name: str = "genetic_model"):
        """
        Inizializza il generatore di modelli genetici.
        
        Args:
            seed: Seme per la generazione pseudocasuale
            complexity_range: Range per la complessità del modello (min, max layers)
            parameter_range: Range per i parametri dimensionali (min, max dimensioni)
            model_base_name: Nome base per il modello generato
        """
        self.rng = np.random.RandomState(seed)
        self.complexity_range = complexity_range
        self.parameter_range = parameter_range
        self.model_base_name = model_base_name
        
        # Default mapping parameters for genome interpretation
        self.gene_value_range = (0, 1)  # Default gene values between 0 and 1
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def generate_from_genome(self, genome: List[float]) -> Dict[str, Any]:
        """
        Genera una configurazione di modello completa a partire da un genoma.
        
        Args:
            genome: Una lista di numeri (es. tra 0 e 1) che codifica il modello
            
        Returns:
            Configurazione JSON completa del modello
        """
        if not genome:
            raise ValueError("Il genoma non può essere vuoto")
        
        # Normalizza il genoma se necessario
        norm_genome = self._normalize_genome(genome)
        
        # Tiene traccia dell'indice corrente nel genoma
        gene_idx = 0
        
        # Struttura di base della configurazione
        config = {
            "meta": {
                "name": f"{self.model_base_name}_{hash(tuple(genome))%10000:04d}",
                "description": "Model generated from genetic code",
                "version": "1.0",
                "genome_hash": hash(tuple(genome)) % (10**10)
            },
            "parameters": [],
            "type_definition": [],
            "model": [],
            "trainings": [],
            "inferences": []
        }
        
        # 1. Determina i parametri base del modello
        gene_idx = self._generate_base_parameters(norm_genome, gene_idx, config)
        
        # 2. Determina i tipi di dato e processori
        gene_idx = self._generate_type_definitions(norm_genome, gene_idx, config)
        
        # 3. Genera i sottomotori del modello
        gene_idx = self._generate_model_structure(norm_genome, gene_idx, config)
        
        # 4. Genera configurazioni di training
        gene_idx = self._generate_training_configs(norm_genome, gene_idx, config)
        
        # 5. Genera configurazioni di inferenza
        gene_idx = self._generate_inference_configs(norm_genome, gene_idx, config)
        
        # Verifica la consistenza del modello e correggi eventuali problemi
        self._ensure_model_consistency(config)
        
        return config
        
    def _normalize_genome(self, genome: List[float]) -> List[float]:
        """Normalizza il genoma nel range [0,1] se necessario."""
        g_min, g_max = min(genome), max(genome)
        
        # Se il genoma è già tra 0 e 1, lo lasciamo così
        if g_min >= 0 and g_max <= 1:
            return genome
            
        # Altrimenti normalizziamo
        return [(g - g_min) / (g_max - g_min) if g_max > g_min else 0.5 for g in genome]
    
    def _get_next_gene(self, genome: List[float], current_idx: int) -> Tuple[float, int]:
        """Ottiene il prossimo valore dal genoma ciclicamente."""
        if current_idx >= len(genome):
            current_idx = 0  # Riparti dall'inizio se abbiamo esaurito il genoma
            
        gene_value = genome[current_idx]
        return gene_value, current_idx + 1
    
    def _get_next_n_genes(self, genome: List[float], current_idx: int, n: int) -> Tuple[List[float], int]:
        """Ottiene i prossimi n valori dal genoma ciclicamente."""
        values = []
        idx = current_idx
        
        for _ in range(n):
            value, idx = self._get_next_gene(genome, idx)
            values.append(value)
            
        return values, idx
    
    def _generate_base_parameters(self, genome: List[float], gene_idx: int, config: Dict[str, Any]) -> int:
        """
        Genera i parametri base del modello.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            config: Configurazione da aggiornare
            
        Returns:
            Nuovo indice nel genoma
        """
        # Determina il numero di parametri base
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        num_params = int(gene * 8) + 3  # Da 3 a 10 parametri base
        
        # Parametri standard che vogliamo sempre avere
        standard_params = [
            {"name": "input_dim", "default": self._map_to_range(genome[gene_idx % len(genome)], 2, 1024)},
            {"name": "hidden_dim", "default": self._map_to_range(genome[(gene_idx + 1) % len(genome)], 4, 512)},
            {"name": "output_dim", "default": self._map_to_range(genome[(gene_idx + 2) % len(genome)], 1, 256)}
        ]
        
        gene_idx += 3  # Abbiamo usato 3 geni per i parametri standard
        
        # Aggiungi i parametri standard
        config["parameters"].extend(standard_params)
        
        # Aggiungi parametri aggiuntivi basati su relazioni con i parametri standard
        for i in range(num_params - 3):  # -3 perché abbiamo già aggiunto 3 parametri standard
            gene, gene_idx = self._get_next_gene(genome, gene_idx)
            gene2, gene_idx = self._get_next_gene(genome, gene_idx)
            
            # Determina se questo parametro è diretto o relativo
            is_relative = gene > 0.7
            
            if is_relative:
                # Parametro relativo rispetto ad un altro
                base_params = ["input_dim", "hidden_dim", "output_dim"]
                base_param_idx = int(gene2 * len(base_params)) % len(base_params)
                base_param = base_params[base_param_idx]
                
                # Tipo di operazione relativa
                operation_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                operations = ["+", "-", "*", "/"]
                op_idx = int(operation_gene * len(operations)) % len(operations)
                operation = operations[op_idx]
                
                # Valore per l'operazione
                value_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                if operation in ["+", "-"]:
                    value = int(value_gene * 128)  # Offset in range [0, 127]
                else:  # "*", "/"
                    value = 0.25 + value_gene * 2.0  # Fattore moltiplicativo/divisivo tra 0.25 e 2.25
                
                # Crea il parametro relativo
                param = {
                    "name": f"derived_dim_{i}",
                    "relative_to": [base_param, operation, value]
                }
            else:
                # Parametro diretto
                dimension_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                dimension = self._map_to_range(dimension_gene, 2, 1024)
                
                param = {
                    "name": f"dim_{i}",
                    "default": dimension
                }
            
            config["parameters"].append(param)
            
        return gene_idx
    
    def _generate_type_definitions(self, genome: List[float], gene_idx: int, config: Dict[str, Any]) -> int:
        """
        Genera le definizioni di tipo per input e output.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            config: Configurazione da aggiornare
            
        Returns:
            Nuovo indice nel genoma
        """
        # Determina il numero di tipi da generare
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        num_types = int(gene * 4) + 1  # Da 1 a 4 tipi
        
        # Possibili tipi di processori
        processor_types = ["text_tokenizer", "image_processor", "vector_normalizer", "identity"]
        output_processor_types = ["token_to_text", "vector_to_class", "identity"]
        
        for i in range(num_types):
            # Determina il tipo di processore
            gene, gene_idx = self._get_next_gene(genome, gene_idx)
            proc_idx = int(gene * len(processor_types)) % len(processor_types)
            processor_type = processor_types[proc_idx]
            
            # Determina il tipo di processore di output
            gene, gene_idx = self._get_next_gene(genome, gene_idx)
            out_proc_idx = int(gene * len(output_processor_types)) % len(output_processor_types)
            output_processor_type = output_processor_types[out_proc_idx]
            
            # Crea la definizione di tipo
            type_def = {
                "name": f"type_{i}",
                "processor": processor_type,
                "output_processor": output_processor_type
            }
            
            # Aggiungi parametri specifici per tipo
            if processor_type == "text_tokenizer":
                gene, gene_idx = self._get_next_gene(genome, gene_idx)
                type_def["max_length"] = int(gene * 512) + 16  # max_length tra 16 e 527
            elif processor_type == "image_processor":
                gene1, gene_idx = self._get_next_gene(genome, gene_idx)
                gene2, gene_idx = self._get_next_gene(genome, gene_idx)
                type_def["size"] = [int(gene1 * 256) + 32, int(gene2 * 256) + 32]  # Dimensioni tra 32 e 287
            
            if output_processor_type == "vector_to_class":
                gene, gene_idx = self._get_next_gene(genome, gene_idx)
                num_classes = int(gene * 10) + 2  # Da 2 a 11 classi
                type_def["classes"] = [f"class_{j}" for j in range(num_classes)]
            
            config["type_definition"].append(type_def)
            
        return gene_idx
    
    def _generate_model_structure(self, genome: List[float], gene_idx: int, config: Dict[str, Any]) -> int:
        """
        Genera la struttura del modello con sottomodelli.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            config: Configurazione da aggiornare
            
        Returns:
            Nuovo indice nel genoma
        """
        # Determina il numero di sottomodelli
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        num_submodels = int(gene * 4) + 1  # Da 1 a 4 sottomodelli
        
        submodel_names = []
        
        for i in range(num_submodels):
            submodel_name = f"submodel_{i}"
            submodel_names.append(submodel_name)
            
            # Determina la complessità del sottomodello (numero di layer)
            gene, gene_idx = self._get_next_gene(genome, gene_idx)
            min_layers, max_layers = self.complexity_range
            num_layers = int(gene * (max_layers - min_layers)) + min_layers
            
            # Crea la configurazione del sottomodello
            submodel = {
                "name": submodel_name,
                "layers": []
            }
            
            # Determina la struttura dei layer
            for j in range(num_layers):
                layer, gene_idx = self._generate_layer(genome, gene_idx, j, num_layers, config["parameters"])
                submodel["layers"].append(layer)
            
            config["model"].append(submodel)
            
        return gene_idx
    
    def _generate_layer(self, genome: List[float], gene_idx: int, layer_idx: int, 
                        total_layers: int, parameters: List[Dict[str, Any]]) -> Tuple[List[Any], int]:
        """
        Genera un singolo layer per il modello.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            layer_idx: Indice del layer corrente
            total_layers: Numero totale di layer
            parameters: Lista dei parametri disponibili
            
        Returns:
            Definizione del layer e nuovo indice nel genoma
        """
        # Scegli il tipo di layer
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        
        # La scelta del layer dipende dalla posizione nella rete
        layer_position = layer_idx / total_layers  # Normalizzato tra 0 e 1
        
        # Gruppi di layer per posizione
        early_layers = ["linear", "conv1d", "conv2d", "embedding"]
        middle_layers = ["linear", "conv1d", "conv2d", "batchnorm1d", "batchnorm2d", "lstm", "gru"]
        late_layers = ["linear", "dropout", "flatten"]
        activation_layers = ["relu", "tanh", "sigmoid", "softmax"]
        pooling_layers = ["maxpool1d", "maxpool2d", "avgpool1d", "avgpool2d"]
        
        # Selezione del gruppo appropriate in base alla posizione
        if layer_position < 0.3:
            eligible_layers = early_layers
        elif layer_position < 0.7:
            eligible_layers = middle_layers + pooling_layers
        else:
            eligible_layers = late_layers
        
        # Determina se questo layer dovrebbe essere un'attivazione
        activation_gene, gene_idx = self._get_next_gene(genome, gene_idx)
        is_activation = activation_gene < 0.4  # 40% di probabilità di essere un'attivazione
        
        if is_activation and layer_idx > 0:  # Le attivazioni non dovrebbero essere il primo layer
            layer_type = activation_layers[int(gene * len(activation_layers)) % len(activation_layers)]
        else:
            layer_type = eligible_layers[int(gene * len(eligible_layers)) % len(eligible_layers)]
        
        # Inizia a costruire il layer
        layer = [layer_type]
        
        # Aggiungi i parametri appropriati in base al tipo di layer
        if layer_type == "linear":
            # Input dimension
            if layer_idx == 0:  # Primo layer, usa un parametro base
                input_param_name = "input_dim"
            else:  # Layer intermedio/finale, usa un parametro derivato o uno base
                param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                eligible_params = [p["name"] for p in parameters]
                input_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
            
            layer.append(input_param_name)
            
            # Output dimension, potrebbe essere il parametro di output se è l'ultimo layer
            if layer_idx == total_layers - 1:  # Ultimo layer
                output_param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                if output_param_gene > 0.5:  # 50% probabilità di usare output_dim
                    layer.append("output_dim")
                else:
                    # Selezione casuale di un altro parametro
                    eligible_params = [p["name"] for p in parameters if p["name"] != input_param_name]
                    if eligible_params:
                        param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                        output_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
                        layer.append(output_param_name)
            else:  # Layer intermedio
                # Selezione casuale di un parametro diverso da quello di input
                eligible_params = [p["name"] for p in parameters if p["name"] != input_param_name]
                if eligible_params:
                    param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    output_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
                    layer.append(output_param_name)
                    
        elif layer_type in ["conv1d", "conv2d"]:
            # Input channels
            if layer_idx == 0:  # Primo layer
                input_param_name = "input_dim"
            else:
                param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                eligible_params = [p["name"] for p in parameters]
                input_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
            
            layer.append(input_param_name)
            
            # Output channels
            param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            eligible_params = [p["name"] for p in parameters if p["name"] != input_param_name]
            if eligible_params:
                output_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
                layer.append(output_param_name)
            
            # Kernel size
            kernel_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            kernel_size = int(kernel_gene * 5) + 1  # kernel size tra 1 e 5
            layer.append(kernel_size)
            
        elif layer_type in ["batchnorm1d", "batchnorm2d"]:
            # Numero di feature
            param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            eligible_params = [p["name"] for p in parameters]
            param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
            layer.append(param_name)
            
        elif layer_type == "dropout":
            # Probabilità di dropout
            dropout_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            dropout_prob = 0.1 + dropout_gene * 0.5  # tra 0.1 e 0.6
            layer.append(dropout_prob)
            
        elif layer_type in ["maxpool1d", "maxpool2d", "avgpool1d", "avgpool2d"]:
            # Kernel size
            kernel_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            kernel_size = int(kernel_gene * 3) + 2  # kernel size tra 2 e 4
            layer.append(kernel_size)
            
            # Stride (opzionale)
            stride_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            if stride_gene > 0.5:  # 50% probabilità di specificare stride
                stride = int(stride_gene * 2) + 1  # stride tra 1 e 2
                layer.append(stride)
                
        elif layer_type == "softmax":
            # Dimensione (opzionale)
            dim_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            if dim_gene > 0.5:  # 50% probabilità di specificare dim
                dim = int(dim_gene * 2) - 1  # dim tra -1 e 0
                layer.append(dim)
                
        elif layer_type in ["lstm", "gru"]:
            # Input size
            if layer_idx == 0:
                input_param_name = "input_dim"
            else:
                param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                eligible_params = [p["name"] for p in parameters]
                input_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
            
            layer.append(input_param_name)
            
            # Hidden size
            param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            eligible_params = [p["name"] for p in parameters if p["name"] != input_param_name]
            if eligible_params:
                hidden_param_name = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
                layer.append(hidden_param_name)
            else:
                layer.append("hidden_dim")
            
            # Num layers (opzionale)
            num_layers_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            if num_layers_gene > 0.7:  # 30% probabilità di specificare num_layers
                num_layers = int(num_layers_gene * 3) + 1  # tra 1 e 3
                layer.append(num_layers)
                
                # Bidirectional (opzionale)
                bidir_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                if bidir_gene > 0.7:  # 30% probabilità di specificare bidirectional
                    layer.append(True)
                
        elif layer_type == "embedding":
            # Num embeddings
            param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            eligible_params = [p["name"] for p in parameters]
            num_embeddings_param = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
            layer.append(num_embeddings_param)
            
            # Embedding dimension
            param_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            eligible_params = [p["name"] for p in parameters if p["name"] != num_embeddings_param]
            if eligible_params:
                embedding_dim_param = eligible_params[int(param_gene * len(eligible_params)) % len(eligible_params)]
                layer.append(embedding_dim_param)
            else:
                layer.append("hidden_dim")
        
        return layer, gene_idx
    
    def _generate_training_configs(self, genome: List[float], gene_idx: int, config: Dict[str, Any]) -> int:
        """
        Genera configurazioni di training.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            config: Configurazione da aggiornare
            
        Returns:
            Nuovo indice nel genoma
        """
        # Determina il numero di configurazioni di training
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        num_trainings = int(gene * 2) + 1  # Da 1 a 2 configurazioni di training
        
        submodel_names = [sm["name"] for sm in config["model"]]
        
        for i in range(num_trainings):
            training_name = f"training_{i}"
            
            # Crea configurazione base
            training_config = {
                "name": training_name,
                "input": [],
                "output": [],
                "losses": []
            }
            
            # Determina l'ottimizzatore
            optimizer_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            optimizer_idx = int(optimizer_gene * len(self.OPTIMIZER_TYPES)) % len(self.OPTIMIZER_TYPES)
            optimizer_type = self.OPTIMIZER_TYPES[optimizer_idx]
            
            # Configura l'ottimizzatore
            lr_gene, gene_idx = self._get_next_gene(genome, gene_idx)
            learning_rate = 10 ** (-4 - lr_gene * 2)  # LR da 10^-4 a 10^-6
            
            training_config["optimizer"] = {
                "type": optimizer_type,
                "params": {
                    "lr": learning_rate
                }
            }
            
            # Aggiungi parametri specifici per ottimizzatore
            if optimizer_type == "sgd":
                momentum_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                training_config["optimizer"]["params"]["momentum"] = 0.8 + momentum_gene * 0.15  # tra 0.8 e 0.95
            
            # Determina quali sottomodelli usare come input/output
            for submodel_name in submodel_names:
                # Decide se usare questo sottomodello come input
                input_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                if input_gene > 0.3:  # 70% probabilità di essere un input
                    # Scegli un tipo da type_definition
                    type_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    if config["type_definition"]:
                        type_idx = int(type_gene * len(config["type_definition"])) % len(config["type_definition"])
                        input_type = config["type_definition"][type_idx]["name"]
                    else:
                        input_type = "default"
                    
                    training_config["input"].append({
                        "name": submodel_name,
                        "type": input_type
                    })
                
                # Decide se usare questo sottomodello come output
                output_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                if output_gene > 0.5:  # 50% probabilità di essere un output
                    # Scegli un tipo da type_definition
                    type_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    if config["type_definition"]:
                        type_idx = int(type_gene * len(config["type_definition"])) % len(config["type_definition"])
                        output_type = config["type_definition"][type_idx]["name"]
                    else:
                        output_type = "default"
                    
                    training_config["output"].append({
                        "name": submodel_name,
                        "type": output_type
                    })
                    
                    # Aggiungi una funzione di loss per questo output
                    loss_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    loss_idx = int(loss_gene * len(self.LOSS_FUNCTIONS)) % len(self.LOSS_FUNCTIONS)
                    loss_type = self.LOSS_FUNCTIONS[loss_idx]
                    
                    weight_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    weight = 0.5 + weight_gene  # tra 0.5 e 1.5
                    
                    training_config["losses"].append({
                        "output": submodel_name,
                        "function": loss_type,
                        "weight": weight
                    })
            
            # Assicurati che ci sia almeno un input e un output
            if not training_config["input"] and submodel_names:
                training_config["input"].append({
                    "name": submodel_names[0],
                    "type": "default"
                })
                
            if not training_config["output"] and submodel_names:
                output_name = submodel_names[-1]
                training_config["output"].append({
                    "name": output_name,
                    "type": "default"
                })
                
                # Aggiungi una funzione di loss di default
                training_config["losses"].append({
                    "output": output_name,
                    "function": "mse",
                    "weight": 1.0
                })
            
            config["trainings"].append(training_config)
            
        return gene_idx
    
    def _generate_inference_configs(self, genome: List[float], gene_idx: int, config: Dict[str, Any]) -> int:
        """
        Genera configurazioni di inferenza.
        
        Args:
            genome: Genoma normalizzato
            gene_idx: Indice corrente nel genoma
            config: Configurazione da aggiornare
            
        Returns:
            Nuovo indice nel genoma
        """
        # Determina il numero di configurazioni di inferenza
        gene, gene_idx = self._get_next_gene(genome, gene_idx)
        num_inferences = int(gene * 2) + 1  # Da 1 a 2 configurazioni di inferenza
        
        submodel_names = [sm["name"] for sm in config["model"]]
        
        for i in range(num_inferences):
            inference_name = f"inference_{i}"
            
            # Crea configurazione base
            inference_config = {
                "name": inference_name,
                "input": [],
                "output": []
            }
            
            # Crea riferimenti tra input e output basati sulle configurazioni di training
            # Per mantenere la coerenza con il training
            if config["trainings"]:
                # Sceglie casualmente una configurazione di training come base
                training_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                training_idx = int(training_gene * len(config["trainings"])) % len(config["trainings"])
                training_config = config["trainings"][training_idx]
                
                # Usa gli stessi input del training
                for input_config in training_config["input"]:
                    inference_config["input"].append(input_config.copy())
                
                # Usa gli stessi output del training
                for output_config in training_config["output"]:
                    inference_config["output"].append(output_config.copy())
            else:
                # Se non ci sono configurazioni di training, crea da zero
                # Determina quali sottomodelli usare come input/output
                for submodel_name in submodel_names:
                    # Decide se usare questo sottomodello come input
                    input_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    if input_gene > 0.3:  # 70% probabilità di essere un input
                        # Scegli un tipo
                        type_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                        if config["type_definition"]:
                            type_idx = int(type_gene * len(config["type_definition"])) % len(config["type_definition"])
                            input_type = config["type_definition"][type_idx]["name"]
                        else:
                            input_type = "default"
                        
                        inference_config["input"].append({
                            "name": submodel_name,
                            "type": input_type
                        })
                    
                    # Decide se usare questo sottomodello come output
                    output_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                    if output_gene > 0.5:  # 50% probabilità di essere un output
                        # Scegli un tipo
                        type_gene, gene_idx = self._get_next_gene(genome, gene_idx)
                        if config["type_definition"]:
                            type_idx = int(type_gene * len(config["type_definition"])) % len(config["type_definition"])
                            output_type = config["type_definition"][type_idx]["name"]
                        else:
                            output_type = "default"
                        
                        inference_config["output"].append({
                            "name": submodel_name,
                            "type": output_type
                        })
            
            # Assicurati che ci sia almeno un input e un output
            if not inference_config["input"] and submodel_names:
                inference_config["input"].append({
                    "name": submodel_names[0],
                    "type": "default"
                })
                
            if not inference_config["output"] and submodel_names:
                inference_config["output"].append({
                    "name": submodel_names[-1],
                    "type": "default"
                })
            
            config["inferences"].append(inference_config)
            
        return gene_idx
    
    def _map_to_range(self, gene_value: float, min_value: int, max_value: int) -> int:
        """Mappa un valore del genoma a un intero in un intervallo specifico."""
        return int(gene_value * (max_value - min_value)) + min_value
    
    def _ensure_model_consistency(self, config: Dict[str, Any]) -> None:
        """
        Assicura che il modello sia coerente, correggendo eventuali problemi.
        """
        # Verifica che ogni sottomodello abbia almeno un layer
        for submodel in config["model"]:
            if not submodel["layers"]:
                # Aggiungi un layer di base
                submodel["layers"] = [["linear", "input_dim", "output_dim"]]
        
        # Verifica che ci sia almeno una configurazione di training
        if not config["trainings"]:
            default_training = {
                "name": "default_training",
                "input": [],
                "output": [],
                "losses": [],
                "optimizer": "adam"
            }
            
            # Aggiungi un input e un output dal primo sottomodello
            if config["model"]:
                first_submodel = config["model"][0]["name"]
                default_training["input"].append({"name": first_submodel, "type": "default"})
                default_training["output"].append({"name": first_submodel, "type": "default"})
                default_training["losses"].append({
                    "output": first_submodel,
                    "function": "mse",
                    "weight": 1.0
                })
            
            config["trainings"].append(default_training)
        
        # Verifica che ci sia almeno una configurazione di inferenza
        if not config["inferences"]:
            default_inference = {
                "name": "default_inference",
                "input": [],
                "output": []
            }
            
            # Copia input e output dalla prima configurazione di training
            if config["trainings"]:
                first_training = config["trainings"][0]
                default_inference["input"] = [inp.copy() for inp in first_training["input"]]
                default_inference["output"] = [out.copy() for out in first_training["output"]]
            # Oppure aggiungi un input e un output dal primo sottomodello
            elif config["model"]:
                first_submodel = config["model"][0]["name"]
                default_inference["input"].append({"name": first_submodel, "type": "default"})
                default_inference["output"].append({"name": first_submodel, "type": "default"})
            
            config["inferences"].append(default_inference)
    
    def get_genome_from_config(self, config: Dict[str, Any]) -> List[float]:
        """
        Estrae un genoma rappresentativo da una configurazione esistente.
        Questa funzione è l'inverso di generate_from_genome.
        
        Args:
            config: Configurazione del modello
            
        Returns:
            Genoma estratto dalla configurazione
        """
        # Questa è una funzione complessa che richiederebbe un'analisi approfondita
        # della configurazione. Per ora, restituiamo un genoma casuale.
        self.logger.warning("get_genome_from_config non è ancora implementata completamente. "
                           "Restituendo un genoma casuale.")
        
        return self.rng.rand(100)  # Genoma casuale di 100 elementi
    
    def mutate_genome(self, genome: List[float], mutation_rate: float = 0.1, 
                     mutation_strength: float = 0.2) -> List[float]:
        """
        Muta un genoma esistente per l'evoluzione genetica.
        
        Args:
            genome: Genoma da mutare
            mutation_rate: Probabilità di mutazione per ciascun gene
            mutation_strength: Intensità della mutazione
            
        Returns:
            Genoma mutato
        """
        mutated_genome = genome.copy()
        
        for i in range(len(mutated_genome)):
            if self.rng.rand() < mutation_rate:
                # Applica una mutazione gaussiana
                delta = self.rng.normal(0, mutation_strength)
                mutated_genome[i] += delta
                
                # Assicurati che rimanga nel range [0, 1]
                mutated_genome[i] = max(0.0, min(1.0, mutated_genome[i]))
        
        return mutated_genome
    
    def crossover_genomes(self, genome1: List[float], genome2: List[float]) -> List[float]:
        """
        Combina due genomi attraverso un crossover genetico.
        
        Args:
            genome1: Primo genoma genitore
            genome2: Secondo genoma genitore
            
        Returns:
            Genoma figlio
        """
        # Assicurati che i genomi abbiano la stessa lunghezza
        min_length = min(len(genome1), len(genome2))
        
        # Punto di crossover casuale
        crossover_point = self.rng.randint(1, min_length - 1)
        
        # Combina i due genomi
        child_genome = genome1[:crossover_point] + genome2[crossover_point:]
        
        return child_genome



class GeneticModelValidator:
    """Classe di utilità per validare e analizzare i modelli genetici."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        Verifica che una configurazione di modello sia valida.
        
        Args:
            config: Configurazione del modello da validare
            
        Returns:
            True se la configurazione è valida, False altrimenti
        """
        # Verifica la presenza di campi obbligatori
        required_fields = ["meta", "parameters", "model", "trainings", "inferences"]
        for field in required_fields:
            if field not in config:
                print(f"Campo mancante: {field}")
                return False
        
        # Verifica che ci sia almeno un sottomodello
        if not config["model"]:
            print("Non ci sono sottomodelli definiti")
            return False
        
        # Verifica che ogni sottomodello abbia un nome e dei layer
        for submodel in config["model"]:
            if "name" not in submodel:
                print("Un sottomodello non ha un nome")
                return False
            if "layers" not in submodel or not submodel["layers"]:
                print(f"Il sottomodello {submodel['name']} non ha layer")
                return False
        
        # Verifica che ci sia almeno una configurazione di training
        if not config["trainings"]:
            print("Non ci sono configurazioni di training")
            return False
        
        # Verifica che le configurazioni di training siano coerenti
        for training in config["trainings"]:
            if "name" not in training:
                print("Una configurazione di training non ha un nome")
                return False
            if "input" not in training or not training["input"]:
                print(f"La configurazione di training {training['name']} non ha input")
                return False
            if "output" not in training or not training["output"]:
                print(f"La configurazione di training {training['name']} non ha output")
                return False
        
        # Verifica che ci sia almeno una configurazione di inferenza
        if not config["inferences"]:
            print("Non ci sono configurazioni di inferenza")
            return False
        
        # Verifica che le configurazioni di inferenza siano coerenti
        for inference in config["inferences"]:
            if "name" not in inference:
                print("Una configurazione di inferenza non ha un nome")
                return False
            if "input" not in inference or not inference["input"]:
                print(f"La configurazione di inferenza {inference['name']} non ha input")
                return False
            if "output" not in inference or not inference["output"]:
                print(f"La configurazione di inferenza {inference['name']} non ha output")
                return False
        
        # Verifica che i parametri siano ben definiti
        for param in config["parameters"]:
            if "name" not in param:
                print("Un parametro non ha un nome")
                return False
            if "default" not in param and "relative_to" not in param:
                print(f"Il parametro {param['name']} non ha valore default né riferimento relativo")
                return False
        
        return True
    
    @staticmethod
    def analyze_model_complexity(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza la complessità di un modello.
        
        Args:
            config: Configurazione del modello da analizzare
            
        Returns:
            Dictionary con metriche di complessità
        """
        analysis = {
            "model_name": config["meta"]["name"],
            "num_submodels": len(config["model"]),
            "total_layers": 0,
            "layer_distribution": {},
            "parameter_count": 0,
            "num_trainings": len(config["trainings"]),
            "num_inferences": len(config["inferences"]),
            "complexity_score": 0
        }
        
        # Conta i layer e la loro distribuzione
        for submodel in config["model"]:
            analysis["total_layers"] += len(submodel["layers"])
            
            for layer in submodel["layers"]:
                layer_type = layer[0]
                if layer_type not in analysis["layer_distribution"]:
                    analysis["layer_distribution"][layer_type] = 0
                analysis["layer_distribution"][layer_type] += 1
        
        # Stima il numero di parametri (approssimazione grossolana)
        param_resolver = ModelParameterResolver(config["parameters"])
        params = param_resolver.get_all_parameters()
        
        # Somma tutti i valori dei parametri come metrica grossolana di complessità parametrica
        for value in params.values():
            if isinstance(value, (int, float)):
                analysis["parameter_count"] += value
        
        # Calcola uno score di complessità
        complexity_score = (
            analysis["num_submodels"] * 5 +
            analysis["total_layers"] * 2 +
            len(analysis["layer_distribution"]) * 3 +
            analysis["num_trainings"] * 2 +
            analysis["num_inferences"] * 1
        )
        analysis["complexity_score"] = complexity_score
        
        return analysis
    
    @staticmethod
    def compare_models(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Confronta due modelli.
        
        Args:
            config1: Prima configurazione del modello
            config2: Seconda configurazione del modello
            
        Returns:
            Dictionary con risultati del confronto
        """
        analysis1 = GeneticModelValidator.analyze_model_complexity(config1)
        analysis2 = GeneticModelValidator.analyze_model_complexity(config2)
        
        comparison = {
            "model1_name": analysis1["model_name"],
            "model2_name": analysis2["model_name"],
            "complexity_diff": analysis1["complexity_score"] - analysis2["complexity_score"],
            "layer_count_diff": analysis1["total_layers"] - analysis2["total_layers"],
            "parameter_count_diff": analysis1["parameter_count"] - analysis2["parameter_count"],
            "shared_layer_types": []
        }
        
        # Trova i tipi di layer in comune
        layer_types1 = set(analysis1["layer_distribution"].keys())
        layer_types2 = set(analysis2["layer_distribution"].keys())
        shared_types = layer_types1.intersection(layer_types2)
        comparison["shared_layer_types"] = list(shared_types)
        
        return comparison

class GeneticEvolutionManager:
    """Gestisce l'evoluzione genetica dei modelli."""
    
    def __init__(self, 
                 population_size: int = 20, 
                 genome_length: int = 100,
                 seed: Optional[int] = None,
                 complexity_range: Tuple[int, int] = (2, 10),
                 parameter_range: Tuple[int, int] = (8, 512)):
        """
        Inizializza il manager di evoluzione genetica.
        
        Args:
            population_size: Dimensione della popolazione
            genome_length: Lunghezza del genoma
            seed: Seme per la generazione pseudocasuale
            complexity_range: Range per la complessità del modello
            parameter_range: Range per i parametri dimensionali
        """
        self.population_size = population_size
        self.genome_length = genome_length
        self.rng = np.random.RandomState(seed)
        
        # Inizializza il generatore di sequenze casuali
        self.sequence_generator = RandomSequenceGenerator(seed)
        
        # Inizializza il generatore di modelli genetici
        self.model_generator = GeneticModelGenerator(
            seed=seed,
            complexity_range=complexity_range,
            parameter_range=parameter_range
        )
        
        # Popolazione corrente di genomi
        self.population = []
        
        # Storico dei fitness
        self.fitness_history = []
        
        # Inizializza la popolazione
        self._initialize_population()
    
    def _initialize_population(self):
        """Inizializza la popolazione con genomi casuali."""
        self.population = []
        
        # Generazione con diverse strategie per massimizzare la diversità
        uniform_count = self.population_size // 3
        normal_count = self.population_size // 3
        chaotic_count = self.population_size - uniform_count - normal_count
        
        # Genera genomi con distribuzione uniforme
        for _ in range(uniform_count):
            genome = self.sequence_generator.uniform_sequence(self.genome_length)
            self.population.append({"genome": genome, "fitness": None})
        
        # Genera genomi con distribuzione normale
        for _ in range(normal_count):
            genome = self.sequence_generator.normal_sequence(self.genome_length)
            self.population.append({"genome": genome, "fitness": None})
        
        # Genera genomi con mappa logistica (caotica)
        for _ in range(chaotic_count):
            genome = self.sequence_generator.chaotic_sequence(self.genome_length)
            self.population.append({"genome": genome, "fitness": None})
    
    def evolve(self, fitness_function, generations: int = 10, 
               mutation_rate: float = 0.1, mutation_strength: float = 0.2,
               elite_count: int = 2):
        """
        Esegue l'evoluzione genetica per un numero specificato di generazioni.
        
        Args:
            fitness_function: Funzione che valuta il fitness di un genoma
            generations: Numero di generazioni
            mutation_rate: Probabilità di mutazione per ciascun gene
            mutation_strength: Intensità della mutazione
            elite_count: Numero di individui d'élite da preservare
            
        Returns:
            Il miglior genoma trovato e il suo fitness
        """
        for generation in range(generations):
            # Valuta il fitness di ogni genoma nella popolazione
            for individual in self.population:
                if individual["fitness"] is None:
                    # Genera config dal genoma
                    config = self.model_generator.generate_from_genome(individual["genome"])
                    # Valuta il fitness
                    individual["fitness"] = fitness_function(config)
            
            # Ordina la popolazione per fitness (dal migliore al peggiore)
            self.population.sort(key=lambda x: x["fitness"], reverse=True)
            
            # Registra la storia del fitness
            best_fitness = self.population[0]["fitness"]
            avg_fitness = sum(i["fitness"] for i in self.population) / len(self.population)
            self.fitness_history.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness
            })
            
            print(f"Generation {generation}: Best fitness = {best_fitness}, Avg fitness = {avg_fitness}")
            
            # Se è l'ultima generazione, termina
            if generation == generations - 1:
                break
            
            # Crea la nuova popolazione
            new_population = []
            
            # Conserva l'élite (i migliori individui)
            for i in range(elite_count):
                new_population.append(self.population[i].copy())
            
            # Genera il resto della popolazione con crossover e mutazione
            while len(new_population) < self.population_size:
                # Selezione dei genitori tramite tournament selection
                parent1 = self._tournament_selection(3)
                parent2 = self._tournament_selection(3)
                
                # Crossover
                child_genome = self.model_generator.crossover_genomes(
                    parent1["genome"], parent2["genome"]
                )
                
                # Mutazione
                child_genome = self.model_generator.mutate_genome(
                    child_genome, mutation_rate, mutation_strength
                )
                
                # Aggiungi alla nuova popolazione
                new_population.append({"genome": child_genome, "fitness": None})
            
            # Sostituisci la vecchia popolazione
            self.population = new_population
        
        # Restituisci il miglior genoma
        return self.population[0]["genome"], self.population[0]["fitness"]
    
    def _tournament_selection(self, tournament_size: int = 3):
        """
        Seleziona un individuo dalla popolazione usando il metodo del torneo.
        
        Args:
            tournament_size: Numero di individui nel torneo
            
        Returns:
            L'individuo selezionato
        """
        # Seleziona individui casuali per il torneo
        tournament = []
        for _ in range(tournament_size):
            idx = self.rng.randint(0, len(self.population))
            tournament.append(self.population[idx])
        
        # Trova il migliore del torneo
        return max(tournament, key=lambda x: x["fitness"])
    
    def get_best_model_config(self):
        """
        Restituisce la configurazione del miglior modello nella popolazione attuale.
        
        Returns:
            Configurazione del miglior modello
        """
        if not self.population:
            raise ValueError("La popolazione è vuota")
        
        # Ordina la popolazione per fitness
        sorted_population = sorted(self.population, key=lambda x: x.get("fitness", float("-inf")), reverse=True)
        
        # Genera la configurazione dal miglior genoma
        best_genome = sorted_population[0]["genome"]
        best_config = self.model_generator.generate_from_genome(best_genome)
        
        return best_config
    
    def evaluate_model(self, model_manager: 'ModelManager', training_name: str, 
                      validation_data: List[Dict[str, Any]], metric: str = 'val_loss',
                      epochs: int = 5, batch_size: int = 32, **kwargs):
        """
        Valuta un modello addestrandolo e misurandone le prestazioni su dati di validazione.
        Può essere usata come fitness function.
        
        Args:
            model_manager: ModelManager per l'addestramento
            training_name: Nome della configurazione di training
            validation_data: Dati di validazione
            metric: Metrica da utilizzare ('val_loss', 'accuracy', ecc.)
            epochs: Numero di epoche di addestramento
            batch_size: Dimensione del batch
            **kwargs: Parametri aggiuntivi per l'addestramento
            
        Returns:
            Un valore di fitness (maggiore è meglio)
        """
        try:
            # Addestra il modello
            history = model_manager.train(
                training_name, 
                validation_data,  # Usa i dati di validazione anche per l'addestramento
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                validation_interval=1,
                **kwargs
            )
            
            # Valuta il modello con l'ultima epoca
            if metric == 'val_loss':
                # Per la loss, minore è meglio, quindi invertiamo
                last_val_loss = history[-1]['val_loss']
                return 1.0 / (1.0 + last_val_loss)
            elif metric in history[-1]:
                # Per altre metriche come accuracy, maggiore è meglio
                return history[-1][metric]
            else:
                # Se la metrica non è disponibile, restituisci un valore di default negativo
                return -1.0
                
        except Exception as e:
            # In caso di errore durante l'addestramento, assegna un fitness molto basso
            print(f"Errore durante la valutazione del modello: {e}")
            return -100.0


# Esempio di utilizzo
if __name__ == "__main__":
    import tempfile
    import os
    import json
    
    print("Generatore Genetico di Modelli PyTorch")
    print("=====================================")
    
    # Crea un generatore di sequenze
    seq_generator = RandomSequenceGenerator(seed=42)
    
    # Genera diversi tipi di sequenze
    uniform_genome = seq_generator.uniform_sequence(100)
    normal_genome = seq_generator.normal_sequence(100)
    chaotic_genome = seq_generator.chaotic_sequence(100)
    prime_genome = seq_generator.prime_based_sequence(100)
    mixed_genome = seq_generator.mixed_sequence(100)
    
    # Inizializza il generatore di modelli
    model_generator = GeneticModelGenerator(
        seed=42,
        complexity_range=(2, 8),
        parameter_range=(8, 256),
        model_base_name="genetic_model"
    )
    
    print("\nGenerazione di modelli da diverse sequenze genomiche...")
    
    # Genera modelli da ciascun genoma
    models = {
        "uniform": model_generator.generate_from_genome(uniform_genome),
        "normal": model_generator.generate_from_genome(normal_genome),
        "chaotic": model_generator.generate_from_genome(chaotic_genome),
        "prime": model_generator.generate_from_genome(prime_genome),
        "mixed": model_generator.generate_from_genome(mixed_genome)
    }
    
    # Analizza ogni modello
    print("\nAnalisi dei modelli generati:")
    for name, model_config in models.items():
        # Verifica che il modello sia valido
        is_valid = GeneticModelValidator.validate_model_config(model_config)
        
        if is_valid:
            # Analizza la complessità
            analysis = GeneticModelValidator.analyze_model_complexity(model_config)
            
            print(f"\nModello '{name}' ({model_config['meta']['name']}):")
            print(f"  Sottomodelli: {analysis['num_submodels']}")
            print(f"  Layers totali: {analysis['total_layers']}")
            print(f"  Distribuzione layers: {analysis['layer_distribution']}")
            print(f"  Configurazioni training: {analysis['num_trainings']}")
            print(f"  Configurazioni inferenza: {analysis['num_inferences']}")
            print(f"  Score complessità: {analysis['complexity_score']}")
            
            # Salva in un file temporaneo
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, f"genetic_model_{name}.json")
            with open(file_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            print(f"  Modello salvato in: {file_path}")
        else:
            print(f"\nModello '{name}' non valido.")
    
    # Confronta due modelli
    print("\nConfronto tra modelli:")
    comparison = GeneticModelValidator.compare_models(models["uniform"], models["chaotic"])
    print(f"  '{comparison['model1_name']}' vs '{comparison['model2_name']}':")
    print(f"  Differenza complessità: {comparison['complexity_diff']}")
    print(f"  Differenza numero layers: {comparison['layer_count_diff']}")
    print(f"  Tipi di layer in comune: {comparison['shared_layer_types']}")
    
    # Esempio di evoluzione genetica
    print("\nEsempio di evoluzione genetica:")
    
    # Funzione di fitness semplificata (in un caso reale, questa valuterebbe il modello)
    def simple_fitness(config):
        """Funzione di fitness semplificata che premia modelli più complessi ma bilanciati."""
        analysis = GeneticModelValidator.analyze_model_complexity(config)
        
        # Premia la varietà di layer e la presenza di più sottomodelli
        variety_score = len(analysis["layer_distribution"]) * 10
        
        # Penalizza complessità eccessiva
        complexity_penalty = max(0, analysis["total_layers"] - 15) * 5
        
        # Premia un equilibrio nel numero di training/inferenza
        balance_score = 10 - abs(analysis["num_trainings"] - analysis["num_inferences"]) * 2
        
        return variety_score + balance_score - complexity_penalty
    
    # Crea il manager di evoluzione
    evolution_manager = GeneticEvolutionManager(
        population_size=10,
        genome_length=80,
        seed=42
    )
    
    # Evolvi per alcune generazioni
    best_genome, best_fitness = evolution_manager.evolve(
        fitness_function=simple_fitness,
        generations=5,
        mutation_rate=0.1,
        mutation_strength=0.2,
        elite_count=2
    )
    
    # Genera il miglior modello
    best_model = model_generator.generate_from_genome(best_genome)
    
    # Analizza il miglior modello
    best_analysis = GeneticModelValidator.analyze_model_complexity(best_model)
    
    print(f"\nMiglior modello dopo evoluzione:")
    print(f"  Nome: {best_model['meta']['name']}")
    print(f"  Fitness: {best_fitness}")
    print(f"  Score complessità: {best_analysis['complexity_score']}")
    print(f"  Layers totali: {best_analysis['total_layers']}")
    print(f"  Distribuzione layers: {best_analysis['layer_distribution']}")
    
    # Esempio di come usare un modello genetico con ModelManager
    print("\nUtilizzo di un modello genetico con ModelManager:")
    
    # Salva la configurazione in un file temporaneo
    temp_config_file = os.path.join(temp_dir, "best_genetic_model.json")
    with open(temp_config_file, 'w') as f:
        json.dump(best_model, f, indent=2)
    
    print(f"  Configurazione salvata in: {temp_config_file}")
    print("  Per utilizzare questo modello:")
    print(f"  1. manager = ModelManager('{temp_config_file}')")
    print("  2. Utilizza i metodi di ModelManager per addestramento e inferenza")
    
    print("\nCompletato!")