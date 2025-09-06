"""
AURA_GENESIS - Advanced Universal Reasoning Architecture
A revolutionary neural network system with biological inspiration
"""

__version__ = "2.0.0"
__author__ = "AURA_GENESIS Team"

# Core imports for easy access
from .core.network import Network
from .core.neuron import Neuron
from .core.thalamus import Thalamus
from .core.hippocampus import Hippocampus
from .core.amygdala import Amygdala

# System imports
from .system.aura_system_manager import AuraSystemManager
from .system.bootloader import AuraBootloader, AuraBootConfig

__all__ = [
    'Network',
    'Neuron', 
    'Thalamus',
    'Hippocampus',
    'Amygdala',
    'AuraSystemManager',
    'AuraBootloader',
    'AuraBootConfig'
]
