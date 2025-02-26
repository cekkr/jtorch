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
              learning_rate: float = 0.001, checkpoint_interval: int = 5):
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
        
        # Setup optimizer and loss functions
        optimizer_type = training_config.get("optimizer", "adam")
        optimizer = self._get_optimizer(optimizer_type, learning_rate)
        
        loss_configs = training_config.get("losses", [])
        loss_functions = self._get_loss_functions(loss_configs)
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_inputs)
                
                # Calculate loss
                loss = 0
                for loss_config in loss_configs:
                    output_name = loss_config["output"]
                    if output_name in outputs and output_name in batch_targets:
                        loss_name = loss_config["function"]
                        weight = loss_config.get("weight", 1.0)
                        loss_fn = loss_functions[loss_name]
                        
                        batch_loss = loss_fn(outputs[output_name], batch_targets[output_name])
                        loss += weight * batch_loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Save checkpoint if needed
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"{training_name}_epoch_{epoch+1}")
        
        # Save final model
        self.save_checkpoint(f"{training_name}_final")
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
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save a model checkpoint."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{checkpoint_name}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'parameters': self.param_resolver.get_all_parameters()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return path
    
    def load_checkpoint(self, checkpoint_path: str):
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
    
    def list_checkpoints(self):
        """List all available checkpoints for this model."""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt'):
                path = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append({
                    'filename': filename,
                    'path': path,
                    'created': datetime.datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M:%S")
                })
        
        return sorted(checkpoints, key=lambda x: x['created'], reverse=True)
    
    def _get_optimizer(self, optimizer_type: str, learning_rate: float) -> torch.optim.Optimizer:
        """Get the optimizer based on the configuration."""
        if optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "adagrad":
            return optim.Adagrad(self.model.parameters(), lr=learning_rate)
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


# Example usage
if __name__ == "__main__":
    # Example JSON path
    config_file = "model_config.json"
    
    # Initialize manager
    manager = ModelManager(config_file)
    
    # Example training data
    training_data = [
        {"sentence": "This is an example sentence", "tokens": [1, 2, 3, 4]},
        {"sentence": "Another example", "tokens": [5, 6]}
    ]
    
    # Train the model
    manager.train("train_sentence_embedding", training_data, epochs=5)
    
    # Save a checkpoint
    checkpoint_path = manager.save_checkpoint("manual_save")
    
    # Load a checkpoint
    manager.load_checkpoint(checkpoint_path)
    
    # Run inference
    inference_input = {"sentence": "Test inference"}
    result = manager.inference("infer_sentence_embedding", inference_input)
    print(result)
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    for checkpoint in checkpoints:
        print(f"{checkpoint['filename']} - {checkpoint['created']}")