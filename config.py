"""
Configuration principale de Kaguya
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class Mode(Enum):
    """Modes de fonctionnement de Kaguya"""
    REALTIME = "realtime"  # Low-latency gaming mode
    QUALITY = "quality"    # High-quality research mode
    REST = "rest"          # Ultra-light wake-on-voice mode
    STUDY = "study"        # Autonomous Wikipedia learning


class VoiceType(Enum):
    """Types de voix disponibles"""
    REALISTIC_HUMAN = "realistic_human"
    ANIME = "anime"


class EmotionStyle(Enum):
    """Styles émotionnels pour la voix et l'avatar"""
    NEUTRAL = "neutral"
    JOYEUX = "joyeux"
    AGACE = "agace"
    TRISTE = "triste"
    EXCITE = "excite"
    FRUSTRE = "frustre"
    DECU = "decu"
    COLERE = "colere"


@dataclass
class HardwareConfig:
    """Configuration matérielle"""
    gpu_name: str = "RTX 4070 Super"
    gpu_vram_gb: int = 12
    cpu_name: str = "i7-13700KF"
    ram_gb: int = 32
    cuda_enabled: bool = True
    
    # Resource limits
    max_vram_usage_gb: float = 10.0  # Garder 2GB de marge
    max_ram_gaming_mode_gb: float = 4.0  # RAM max en mode gaming
    max_ram_quality_mode_gb: float = 16.0


@dataclass
class AudioConfig:
    """Configuration audio"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    
    # STT settings
    stt_model: str = "openai/whisper-large-v3"  # Local Whisper
    stt_device: str = "cuda"
    stt_language: str = "fr"
    
    # TTS settings
    tts_model_realistic: str = "facebook/mms-tts-fra"
    tts_model_anime: str = "rvc-anime-fr"  # À configurer
    tts_device: str = "cuda"
    
    # Voice activity detection
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 500
    
    # Diarization & voiceprint
    enable_diarization: bool = True
    enable_voiceprint: bool = True
    voiceprint_threshold: float = 0.75


@dataclass
class MemoryConfig:
    """Configuration de la mémoire"""
    # Chemins de stockage
    memory_dir: str = "./data/memory"
    short_term_db: str = "short_term.db"
    long_term_db: str = "long_term.db"
    knowledge_db: str = "knowledge.db"
    
    # Limites
    short_term_max_entries: int = 100
    long_term_max_entries: int = 10000
    knowledge_max_entries: int = 50000
    
    # Decay & priority
    enable_decay: bool = True
    decay_factor_days: float = 120.0  # ~4 mois
    min_priority_threshold: float = 0.1
    
    # Refresh cycle
    knowledge_refresh_days: int = 120


@dataclass
class EmbodimentConfig:
    """Configuration du VTuber embodiment"""
    # Display
    display_index: int = 1  # 2e écran
    window_width: int = 1920
    window_height: int = 1080
    fullscreen: bool = True
    fps_target: int = 60
    
    # Avatar
    model_path: str = "./assets/models/kaguya.vrm"  # VRM model
    
    # Animation states
    enable_lip_sync: bool = True
    enable_gaze: bool = True
    enable_idle_behaviors: bool = True
    
    # Behaviors
    idle_behavior_interval_s: float = 30.0
    bored_threshold_s: float = 300.0  # 5 minutes
    attention_seeking_threshold_s: float = 600.0  # 10 minutes


@dataclass
class PresenceConfig:
    """Configuration détection de présence"""
    enable_webcam: bool = False  # Optionnel
    webcam_index: int = 0
    
    # Règles d'interaction
    require_headset_confirmation: bool = True
    interrupt_max_attempts: int = 3
    interrupt_wait_between_s: float = 5.0
    
    # Face detection (si webcam activée)
    enable_face_detection: bool = True
    enable_face_recognition: bool = False
    attention_detection_threshold: float = 0.7


@dataclass
class StudyConfig:
    """Configuration mode Study"""
    enable_autonomous_study: bool = True
    study_wikipedia_only: bool = True
    wikipedia_language: str = "fr"
    
    # Scheduling
    study_idle_time_threshold_s: float = 300.0  # Démarre après 5min d'inactivité
    study_max_duration_s: float = 3600.0  # Max 1h de study par session
    
    # Checkpoints
    checkpoint_interval_s: float = 300.0  # Checkpoint tous les 5 minutes
    checkpoint_dir: str = "./data/study/checkpoints"
    
    # Learning
    max_articles_per_session: int = 10
    summary_compression_ratio: float = 0.1  # Compresser à 10% du texte original


@dataclass
class InternetConfig:
    """Configuration accès Internet"""
    enable_internet: bool = True
    
    # Whitelist
    default_whitelist: List[str] = None
    allow_custom_whitelist: bool = True
    
    def __post_init__(self):
        if self.default_whitelist is None:
            self.default_whitelist = [
                "fr.wikipedia.org",
                "en.wikipedia.org",
                "*.wikimedia.org"
            ]


@dataclass
class WakeConfig:
    """Configuration wake word et rest mode"""
    wake_word: str = "kaguya"
    wake_confidence_threshold: float = 0.8
    
    # Rest mode
    rest_mode_cpu_limit_percent: float = 5.0
    rest_mode_ram_limit_mb: float = 500.0


@dataclass
class KaguayConfig:
    """Configuration globale de Kaguya"""
    # Sous-configurations
    hardware: HardwareConfig = None
    audio: AudioConfig = None
    memory: MemoryConfig = None
    embodiment: EmbodimentConfig = None
    presence: PresenceConfig = None
    study: StudyConfig = None
    internet: InternetConfig = None
    wake: WakeConfig = None
    
    def __post_init__(self):
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.embodiment is None:
            self.embodiment = EmbodimentConfig()
        if self.presence is None:
            self.presence = PresenceConfig()
        if self.study is None:
            self.study = StudyConfig()
        if self.internet is None:
            self.internet = InternetConfig()
        if self.wake is None:
            self.wake = WakeConfig()
    
    # Mode par défaut
    default_mode: Mode = Mode.REALTIME
    
    # Voix par défaut
    default_voice: VoiceType = VoiceType.ANIME
    default_emotion: EmotionStyle = EmotionStyle.NEUTRAL
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # User
    user_name: str = "Maître"  # Personnalisable
    
    def save(self, path: str = "./config/kaguya_config.json"):
        """Sauvegarder la configuration"""
        import json
        from dataclasses import asdict
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str = "./config/kaguya_config.json"):
        """Charger la configuration"""
        import json
        
        if not os.path.exists(path):
            # Créer config par défaut
            config = cls()
            config.save(path)
            return config
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Reconstruire les objets (simplifié, à améliorer)
        return cls(**config_dict)


# Instance globale
config = KaguayConfig()
