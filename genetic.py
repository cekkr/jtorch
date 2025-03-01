# Example usage
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
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
import torch.nn as nn
import logging

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