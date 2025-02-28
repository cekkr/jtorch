#!/usr/bin/env python3
"""
WebGPU Accelerator - Un wrapper Python per WebGPU Dawn
Fornisce accelerazione GPU per calcoli matematici utilizzando Dawn
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional
import ctypes
from enum import Enum

class GPUBackend(Enum):
    """Enumerazione dei backend supportati da Dawn"""
    D3D12 = "d3d12"
    METAL = "metal"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    AUTO = "auto"  # Selezione automatica del backend migliore

class DawnManager:
    """Gestisce il caricamento e l'inizializzazione di Dawn WebGPU"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DawnManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inizializza il backend Dawn"""
        self.dawn_lib = None
        self._detect_platform()
        self._load_dawn_library()
    
    def _detect_platform(self):
        """Rileva sistema operativo, architettura CPU e GPU"""
        self.os_name = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # Mappa l'architettura CPU per nomi comuni
        if self.arch in ["x86_64", "amd64"]:
            self.arch = "x64"
        elif self.arch in ["aarch64", "arm64"]:
            self.arch = "arm64"
        
        # Determina il tipo di GPU e il relativo backend
        self.gpu_type = self._detect_gpu_type()
        
        # Percorsi per i binari precompilati
        self.bin_dir = Path(__file__).parent / "bin" / f"{self.os_name}-{self.arch}-{self.gpu_type}"
        self.build_dir = Path(__file__).parent / "build"
        
    def _detect_gpu_type(self):
        """Rileva il tipo di GPU disponibile nel sistema"""
        # Inizialmente assumiamo un backend generico
        gpu_type = "generic"
        
        if self.os_name == "darwin":
            # Su macOS, utilizziamo Metal Performance Shaders (MPS)
            gpu_type = "mps"
        else:
            # Per Linux e Windows dobbiamo rilevare CUDA o ROCm
            try:
                # Verifica la presenza di CUDA
                if self._check_cuda_available():
                    gpu_type = "cuda"
                # Verifica la presenza di ROCm
                elif self._check_rocm_available():
                    gpu_type = "rocm"
            except Exception as e:
                print(f"Avviso durante il rilevamento GPU: {e}")
                pass
                
        return gpu_type
    
    def _check_cuda_available(self):
        """Verifica se CUDA è disponibile nel sistema"""
        # Verifica l'esistenza di librerie CUDA
        cuda_paths = [
            "/usr/local/cuda/lib64/libcudart.so",  # Linux
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin\\cudart64_*.dll",  # Windows
        ]
        
        # Verifica se nvidia-smi è disponibile
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        
        # Verifica le librerie
        for path in cuda_paths:
            if self.os_name == "windows" and "*" in path:
                # Per Windows, consideriamo pattern con wildcard
                import glob
                if glob.glob(path):
                    return True
            elif os.path.exists(path):
                return True
                
        return False
    
    def _check_rocm_available(self):
        """Verifica se ROCm è disponibile nel sistema"""
        # Verifica l'esistenza di librerie ROCm
        rocm_paths = [
            "/opt/rocm/lib/librocm_smi64.so",  # Linux
            "/opt/rocm/bin/rocm-smi",  # Utility ROCm
        ]
        
        # Verifica se rocm-smi è disponibile
        try:
            result = subprocess.run(
                ["rocm-smi"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        
        # Verifica le librerie
        for path in rocm_paths:
            if os.path.exists(path):
                return True
                
        return False
    
    def _load_dawn_library(self):
        """Carica la libreria Dawn, compilandola se necessario"""
        # Verifica se esiste una versione precompilata
        lib_name = self._get_library_name()
        lib_path = self.bin_dir / lib_name
        
        if lib_path.exists():
            # Usa la versione precompilata
            self.dawn_lib = self._load_library(lib_path)
        else:
            # Compila Dawn per l'hardware corrente
            self._compile_dawn()
            # Carica la libreria appena compilata
            lib_path = self.build_dir / lib_name
            if lib_path.exists():
                self.dawn_lib = self._load_library(lib_path)
            else:
                raise RuntimeError(f"Impossibile trovare o compilare Dawn per {self.os_name}-{self.arch}")
    
    def _get_library_name(self) -> str:
        """Restituisce il nome della libreria in base al sistema operativo"""
        if self.os_name == "windows":
            return "dawn.dll"
        elif self.os_name == "darwin":
            return "libdawn.dylib"
        else:  # Linux e altri Unix
            return "libdawn.so"
    
    def _load_library(self, path: Path):
        """Carica la libreria dinamica"""
        try:
            if self.os_name == "windows":
                return ctypes.WinDLL(str(path))
            else:
                return ctypes.CDLL(str(path))
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento della libreria Dawn: {e}")
    
    def _compile_dawn(self):
        """Compila Dawn per l'hardware corrente considerando il tipo di GPU"""
        print(f"Compilazione di Dawn per {self.os_name}-{self.arch} con supporto {self.gpu_type}...")
        
        # Assicurati che la directory di build esista
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clona il repository Dawn se non esiste
        dawn_src = Path(__file__).parent / "third_party" / "dawn"
        if not dawn_src.exists():
            subprocess.run([
                "git", "clone", 
                "https://dawn.googlesource.com/dawn", 
                str(dawn_src)
            ], check=True)
            
            # Inizializza e aggiorna i submodule
            subprocess.run([
                "git", "submodule", "update", "--init", "--recursive"
            ], cwd=str(dawn_src), check=True)
        
        # Opzioni di base per CMake
        cmake_args = [
            "cmake", "-B", str(self.build_dir), 
            "-S", str(dawn_src),
            "-DDAWN_BUILD_EXAMPLES=OFF",
            "-DDAWN_BUILD_TESTS=OFF",
            "-DDAWN_ENABLE_OPENGLES=ON"
        ]
        
        # Opzioni specifiche per il tipo di GPU
        if self.gpu_type == "cuda":
            # Aggiungi opzioni per abilitare il supporto CUDA
            cmake_args.extend([
                "-DDAWN_ENABLE_CUDA=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=all"
            ])
            
            # Cerca di individuare CUDA toolkit
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}")
                
        elif self.gpu_type == "rocm":
            # Aggiungi opzioni per abilitare il supporto ROCm/HIP
            cmake_args.extend([
                "-DDAWN_ENABLE_ROCM=ON"
            ])
            
            # Cerca di individuare il path di ROCm
            rocm_path = "/opt/rocm"  # Path predefinito
            if os.path.exists(rocm_path):
                cmake_args.append(f"-DROCM_PATH={rocm_path}")
                
        elif self.gpu_type == "mps" and self.os_name == "darwin":
            # Per macOS con Metal, abilita specificamente il backend Metal
            cmake_args.extend([
                "-DDAWN_ENABLE_METAL=ON",
                "-DDAWN_USE_BUILT_DXC=ON"  # Usiamo il compilatore HLSL integrato
            ])
        
        # Imposta l'architettura di compilazione
        if self.arch == "arm64":
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64" if self.os_name == "darwin" else "-DCMAKE_SYSTEM_PROCESSOR=aarch64")
        
        print(f"Configurazione CMake: {' '.join(cmake_args)}")
        subprocess.run(cmake_args, check=True)
        
        # Compila
        build_args = [
            "cmake", "--build", str(self.build_dir), 
            "--config", "Release", 
            "--parallel"
        ]
        
        print(f"Esecuzione build: {' '.join(build_args)}")
        subprocess.run(build_args, check=True)
        
        # Copia i binari compilati nella directory dei binari precompilati
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Copia le librerie necessarie nella directory dei binari
        lib_name = self._get_library_name()
        src_lib = self.build_dir / lib_name
        if src_lib.exists():
            shutil.copy(src_lib, self.bin_dir / lib_name)
            print(f"Libreria copiata in {self.bin_dir / lib_name}")
        
        print("Compilazione completata.")
    
    def get_library(self):
        """Restituisce la libreria Dawn caricata"""
        if self.dawn_lib is None:
            raise RuntimeError("Dawn non è stato inizializzato correttamente")
        return self.dawn_lib


class WebGPUDevice:
    """Rappresenta un dispositivo WebGPU"""
    
    def __init__(self, backend: GPUBackend = GPUBackend.AUTO):
        self.dawn_manager = DawnManager()
        self.dawn_lib = self.dawn_manager.get_library()
        self.gpu_type = self.dawn_manager.gpu_type
        
        # Se il backend è AUTO, selezioniamo automaticamente in base al tipo di GPU
        if backend == GPUBackend.AUTO:
            self.backend = self._select_optimal_backend()
        else:
            self.backend = backend
            
        self._initialize_device()
    
    def _select_optimal_backend(self) -> GPUBackend:
        """Seleziona il backend ottimale in base al tipo di GPU rilevato"""
        gpu_type = self.gpu_type
        os_name = self.dawn_manager.os_name
        
        if gpu_type == "cuda":
            # Per CUDA, Vulkan è generalmente la scelta migliore
            return GPUBackend.VULKAN
        elif gpu_type == "rocm":
            # Per ROCm, Vulkan è la scelta migliore su Linux
            return GPUBackend.VULKAN
        elif gpu_type == "mps" and os_name == "darwin":
            # Per macOS con MPS, Metal è l'unica opzione reale
            return GPUBackend.METAL
        elif os_name == "windows":
            # Su Windows senza GPU specificate, D3D12 è generalmente migliore
            return GPUBackend.D3D12
        else:
            # Fallback a Vulkan per la maggior parte delle altre configurazioni
            return GPUBackend.VULKAN
    
    def _initialize_device(self):
        """Inizializza il dispositivo WebGPU"""
        # Chiama le API Dawn per inizializzare il dispositivo
        # Questo è un esempio semplificato, l'implementazione reale
        # utilizzerà le funzioni effettive dalle librerie Dawn
        
        # Impostazione del backend
        backend_str = self.backend.value
        
        print(f"Inizializzazione dispositivo WebGPU con backend: {backend_str} per GPU: {self.gpu_type}")
        
        # Opzioni specifiche per il tipo di GPU
        adapter_options = {}
        
        if self.gpu_type == "cuda" and backend_str == "vulkan":
            # Per CUDA con Vulkan, possiamo aggiungere opzioni specifiche
            adapter_options["preferGPUAdapter"] = "NVIDIA"
        elif self.gpu_type == "rocm" and backend_str == "vulkan":
            # Per ROCm con Vulkan, preferiamo adattatori AMD
            adapter_options["preferGPUAdapter"] = "AMD"
        
        # Inizializzazione del dispositivo (pseudo-codice, da implementare con API Dawn)
        # self.device = self.dawn_lib.CreateDevice(backend_str, adapter_options)
        # if not self.device:
        #    raise RuntimeError(f"Impossibile creare dispositivo WebGPU con backend {backend_str}")
        
        # Per ora usiamo un segnaposto
        self.device = f"device_{backend_str}_{self.gpu_type}"
        
        # Stampa informazioni sul dispositivo
        print(f"Dispositivo WebGPU inizializzato: {self.device}")
        
        # Recupera e memorizza le capacità del dispositivo
        self.compute_capabilities = self._get_compute_capabilities()
    
    def _get_compute_capabilities(self):
        """Recupera le capacità di calcolo del dispositivo"""
        # In un'implementazione reale, questa funzione interrogherebbe
        # il dispositivo per ottenere informazioni sulle sue capacità
        
        # Ritorna un dizionario con informazioni base
        return {
            "max_compute_workgroups": [65535, 65535, 65535],
            "max_compute_workgroup_size": [1024, 1024, 64],
            "max_compute_invocations_per_workgroup": 1024,
            "max_storage_buffer_binding_size": 1 << 30,  # 1GB
            "max_buffer_size": 1 << 30,  # 1GB
            "backend": self.backend.value,
            "gpu_type": self.gpu_type
        }
    
    def create_buffer(self, data: np.ndarray, usage: str = "storage,copy_dst,copy_src"):
        """Crea un buffer WebGPU dal dato NumPy"""
        # Implementazione segnaposto
        return WebGPUBuffer(self, data, usage)
    
    def create_compute_pipeline(self, shader_code: str):
        """Crea una pipeline di calcolo dal codice shader WGSL"""
        # Implementazione segnaposto
        return WebGPUComputePipeline(self, shader_code)


class WebGPUBuffer:
    """Buffer di memoria GPU per WebGPU"""
    
    def __init__(self, device: WebGPUDevice, data: np.ndarray, usage: str):
        self.device = device
        self.data = data
        self.usage = usage
        self.size = data.nbytes
        self.dtype = data.dtype
        self.shape = data.shape
        
        # Creazione del buffer (pseudo-codice)
        # self.buffer = device.dawn_lib.CreateBuffer(size=self.size, usage=usage)
        # self.write(data)
        
        # Per ora usiamo un segnaposto
        self.buffer = f"buffer_{id(self)}"
    
    def write(self, data: np.ndarray):
        """Scrive dati nel buffer"""
        # Implementazione segnaposto
        self.data = data
    
    def read(self) -> np.ndarray:
        """Legge i dati dal buffer in un array NumPy"""
        # Implementazione segnaposto
        return self.data


class WebGPUComputePipeline:
    """Pipeline di calcolo WebGPU"""
    
    def __init__(self, device: WebGPUDevice, shader_code: str):
        self.device = device
        self.shader_code = shader_code
        
        # Compilazione dello shader (pseudo-codice)
        # self.shader_module = device.dawn_lib.CreateShaderModule(shader_code)
        # self.pipeline = device.dawn_lib.CreateComputePipeline(self.shader_module)
        
        # Per ora usiamo un segnaposto
        self.pipeline = f"pipeline_{id(self)}"
    
    def dispatch(self, x: int, y: int = 1, z: int = 1, buffers: List[WebGPUBuffer] = None):
        """Esegue il kernel di calcolo"""
        # Implementazione segnaposto
        # In una implementazione reale, qui si associano i buffer e si esegue il kernel
        pass


class GPUArray:
    """Array GPU, simile a un array NumPy ma eseguito su GPU"""
    
    def __init__(self, data: np.ndarray, device: WebGPUDevice = None):
        if device is None:
            self.device = WebGPUDevice()
        else:
            self.device = device
        
        self.shape = data.shape
        self.dtype = data.dtype
        self.buffer = self.device.create_buffer(data)
    
    def to_numpy(self) -> np.ndarray:
        """Converte l'array GPU in array NumPy"""
        return self.buffer.read()
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, device: WebGPUDevice = None) -> 'GPUArray':
        """Crea un GPUArray da un array NumPy"""
        return cls(array, device)
    
    def __matmul__(self, other: 'GPUArray') -> 'GPUArray':
        """Implementa la moltiplicazione matriciale @"""
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("La moltiplicazione matriciale richiede matrici 2D")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Dimensioni incompatibili per moltiplicazione matriciale: {self.shape} e {other.shape}")
        
        # Dimensioni dell'output
        m, n = self.shape[0], other.shape[1]
        
        # Shader WGSL per moltiplicazione matriciale
        shader_code = f"""
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let row = global_id.x;
            let col = global_id.y;
            
            if (row >= {m} || col >= {n}) {{
                return;
            }}
            
            var sum = 0.0;
            let k = {self.shape[1]};
            
            for (var i = 0u; i < k; i = i + 1) {{
                sum = sum + a[row * k + i] * b[i * {n} + col];
            }}
            
            c[row * {n} + col] = sum;
        }}
        """
        
        # Creazione della pipeline
        pipeline = self.device.create_compute_pipeline(shader_code)
        
        # Creazione del buffer di output
        result_array = np.zeros((m, n), dtype=np.float32)
        output_buffer = self.device.create_buffer(result_array)
        
        # Esecuzione del kernel
        pipeline.dispatch(
            (m + 15) // 16, (n + 15) // 16, 
            buffers=[self.buffer, other.buffer, output_buffer]
        )
        
        # Creazione dell'array risultante
        result = GPUArray.__new__(GPUArray)
        result.device = self.device
        result.shape = (m, n)
        result.dtype = np.float32
        result.buffer = output_buffer
        
        return result
    
    def __add__(self, other: 'GPUArray') -> 'GPUArray':
        """Implementa l'addizione elemento per elemento"""
        if self.shape != other.shape:
            raise ValueError(f"Forme incompatibili per addizione: {self.shape} e {other.shape}")
        
        # Numero totale di elementi
        n = np.prod(self.shape)
        
        # Shader WGSL per addizione
        shader_code = f"""
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            
            if (idx >= {n}) {{
                return;
            }}
            
            c[idx] = a[idx] + b[idx];
        }}
        """
        
        # Creazione della pipeline
        pipeline = self.device.create_compute_pipeline(shader_code)
        
        # Creazione del buffer di output
        result_array = np.zeros(self.shape, dtype=np.float32)
        output_buffer = self.device.create_buffer(result_array)
        
        # Esecuzione del kernel
        pipeline.dispatch(
            (n + 255) // 256, 
            buffers=[self.buffer, other.buffer, output_buffer]
        )
        
        # Creazione dell'array risultante
        result = GPUArray.__new__(GPUArray)
        result.device = self.device
        result.shape = self.shape
        result.dtype = np.float32
        result.buffer = output_buffer
        
        return result


# Esempio di utilizzo
def main():
    # Creazione di matrici di test
    a = np.random.rand(1024, 1024).astype(np.float32)
    b = np.random.rand(1024, 1024).astype(np.float32)
    
    # Esecuzione su CPU con NumPy
    import time
    print("Esecuzione su CPU...")
    t0 = time.time()
    cpu_result = a @ b
    cpu_time = time.time() - t0
    print(f"CPU: {cpu_time:.4f} secondi")
    
    # Esecuzione su GPU con WebGPU
    print("Esecuzione su GPU...")
    device = WebGPUDevice()
    gpu_a = GPUArray.from_numpy(a, device)
    gpu_b = GPUArray.from_numpy(b, device)
    
    t0 = time.time()
    gpu_result = gpu_a @ gpu_b
    result_np = gpu_result.to_numpy()
    gpu_time = time.time() - t0
    print(f"GPU: {gpu_time:.4f} secondi")
    
    # Verifica della correttezza
    max_diff = np.max(np.abs(cpu_result - result_np))
    print(f"Differenza massima: {max_diff}")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    main()