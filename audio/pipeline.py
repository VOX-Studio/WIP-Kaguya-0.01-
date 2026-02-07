"""
Pipeline audio pour Kaguya
Gère STT (Speech-to-Text), TTS (Text-to-Speech) et Wake Word Detection
"""

import torch
import numpy as np
import sounddevice as sd
import pyaudio
import struct
import threading
import queue
import time
from typing import Optional, Callable
import logging

# Transformers pour Whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Coqui TTS pour la synthèse vocale
from TTS.api import TTS

# Porcupine pour wake word detection
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logging.warning("Porcupine non disponible. Wake word detection désactivée.")

logger = logging.getLogger(__name__)


class SpeechToText:
    """Reconnaissance vocale avec Whisper"""
    
    def __init__(self, config: dict):
        self.config = config
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    def load(self, model_size: str = "large-v3"):
        """
        Charge le modèle Whisper
        
        Args:
            model_size: Taille du modèle (tiny, base, small, medium, large-v3)
        """
        try:
            logger.info(f"Chargement du modèle Whisper {model_size}...")
            model_name = f"openai/whisper-{model_size}"
            
            # CORRECTION: Utiliser WhisperProcessor au lieu de AutoProcessor
            self.processor = WhisperProcessor.from_pretrained(
                model_name,
                language="fr",
                task="transcribe"
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16  # Optimisation pour RTX
            )
            
            self.model.to(self.device)
            self.is_loaded = True
            
            logger.info(f"✓ Modèle Whisper chargé sur {self.device}")
            
        except Exception as e:
            logger.error(f"✗ Erreur chargement STT: {e}")
            raise
    
    def load_light_model(self):
        """Charge un modèle léger pour le mode gaming"""
        self.load(model_size="base")
    
    def unload(self):
        """Décharge le modèle pour libérer la mémoire"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("Modèle STT déchargé")
    
    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcrit l'audio en texte
        
        Args:
            audio_array: Audio sous forme de numpy array
            sample_rate: Taux d'échantillonnage (16kHz pour Whisper)
            
        Returns:
            Texte transcrit
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle STT n'est pas chargé")
        
        try:
            # Prétraitement de l'audio
            inputs = self.processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features
            
            inputs = inputs.to(self.device)
            
            # Génération de la transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs,
                    language="fr",
                    task="transcribe"
                )
            
            # Décodage
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"✗ Erreur transcription: {e}")
            return ""


class TextToSpeech:
    """Synthèse vocale avec Coqui TTS"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.sample_rate = config.get('tts', {}).get('sample_rate', 22050)
        
    def load(self, model_name: str = None):
        """
        Charge le modèle TTS
        
        Args:
            model_name: Nom du modèle (utilise la config par défaut si None)
        """
        try:
            logger.info("Chargement du modèle TTS...")
            
            if model_name is None:
                model_name = self.config.get('tts', {}).get(
                    'model',
                    'tts_models/multilingual/multi-dataset/your_tts'
                )
            
            # Chargement du modèle Coqui TTS
            self.model = TTS(
                model_name=model_name,
                progress_bar=False,
                gpu=(self.device == "cuda")
            )
            
            self.is_loaded = True
            logger.info(f"✓ Modèle TTS chargé sur {self.device}")
            
        except Exception as e:
            logger.error(f"✗ Erreur chargement TTS: {e}")
            raise
    
    def unload(self):
        """Décharge le modèle TTS"""
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("Modèle TTS déchargé")
    
    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        speed: float = 1.0,
        speaker_wav: str = None
    ) -> Optional[np.ndarray]:
        """
        Synthétise le texte en audio
        
        Args:
            text: Texte à synthétiser
            emotion: Émotion de la voix (neutral, happy, sad, angry)
            speed: Vitesse de parole (0.5 à 2.0)
            speaker_wav: Chemin vers un fichier audio de référence (clonage de voix)
            
        Returns:
            Audio sous forme de numpy array ou None en cas d'erreur
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle TTS n'est pas chargé")
        
        try:
            # Utilisation de la voix de référence depuis la config si non fournie
            if speaker_wav is None:
                speaker_wav = self.config.get('tts', {}).get('voice_path')
            
            # Génération de l'audio
            wav = self.model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language="fr",
                speed=speed
            )
            
            # Conversion en numpy array
            audio_array = np.array(wav, dtype=np.float32)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"✗ Erreur synthèse TTS: {e}")
            return None
    
    def play(self, audio_array: np.ndarray, sample_rate: int = None):
        """
        Joue l'audio généré
        
        Args:
            audio_array: Audio à jouer
            sample_rate: Taux d'échantillonnage (utilise self.sample_rate par défaut)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            sd.play(audio_array, sample_rate)
            sd.wait()  # Attend la fin de la lecture
        except Exception as e:
            logger.error(f"✗ Erreur lecture audio: {e}")
    
    def save(self, audio_array: np.ndarray, filepath: str, sample_rate: int = None):
        """
        Sauvegarde l'audio dans un fichier
        
        Args:
            audio_array: Audio à sauvegarder
            filepath: Chemin du fichier de sortie
            sample_rate: Taux d'échantillonnage
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(filepath, sample_rate, audio_array)
            logger.info(f"Audio sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"✗ Erreur sauvegarde audio: {e}")


class WakeWordDetector:
    """Détection de wake word avec Porcupine"""
    
    def __init__(self, config: dict):
        self.config = config
        self.porcupine = None
        self.is_loaded = False
        self.audio_stream = None
        self.is_listening = False
        self.callback = None
        
    def load(self):
        """Charge le détecteur de wake word"""
        if not PORCUPINE_AVAILABLE:
            logger.warning("Porcupine non disponible, wake word désactivé")
            return
        
        try:
            logger.info("Chargement du wake word detector...")
            
            # Configuration du wake word
            keywords = self.config.get('wake_word', {}).get('keywords', ['computer'])
            sensitivity = self.config.get('wake_word', {}).get('sensitivity', 0.5)
            
            # Création de l'instance Porcupine
            self.porcupine = pvporcupine.create(
                keywords=keywords,
                sensitivities=[sensitivity] * len(keywords)
            )
            
            self.is_loaded = True
            logger.info(f"✓ Wake word detector chargé (keywords: {keywords})")
            
        except Exception as e:
            logger.error(f"✗ Erreur chargement wake word: {e}")
            raise
    
    def start_listening(self, callback: Callable):
        """
        Démarre l'écoute du wake word
        
        Args:
            callback: Fonction appelée lors de la détection du wake word
        """
        if not self.is_loaded:
            logger.warning("Wake word detector non chargé")
            return
        
        self.callback = callback
        self.is_listening = True
        
        # Démarrage du thread d'écoute
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("👂 Écoute du wake word démarrée")
    
    def stop_listening(self):
        """Arrête l'écoute du wake word"""
        self.is_listening = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        logger.info("🔇 Écoute du wake word arrêtée")
    
    def _listen_loop(self):
        """Boucle d'écoute du wake word"""
        pa = pyaudio.PyAudio()
        
        try:
            self.audio_stream = pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            while self.is_listening:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    logger.info(f"🎯 Wake word détecté! (index: {keyword_index})")
                    if self.callback:
                        self.callback(keyword_index)
                        
        except Exception as e:
            logger.error(f"Erreur dans la boucle d'écoute: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.close()
            pa.terminate()
    
    def unload(self):
        """Décharge le détecteur"""
        self.stop_listening()
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        self.is_loaded = False
        logger.info("Wake word detector déchargé")


class AudioRecorder:
    """Enregistrement audio pour la reconnaissance vocale"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
    def start_recording(self):
        """Démarre l'enregistrement"""
        self.is_recording = True
        threading.Thread(target=self._record_loop, daemon=True).start()
        logger.info("🎙️ Enregistrement démarré")
    
    def stop_recording(self) -> np.ndarray:
        """
        Arrête l'enregistrement et retourne l'audio
        
        Returns:
            Audio enregistré sous forme de numpy array
        """
        self.is_recording = False
        time.sleep(0.1)  # Laisse le temps au thread de finir
        
        # Récupération de tous les chunks audio
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if audio_data:
            audio_array = np.concatenate(audio_data)
            logger.info(f"🎙️ Enregistrement arrêté ({len(audio_array)/self.sample_rate:.2f}s)")
            return audio_array
        else:
            return np.array([], dtype=np.float32)
    
    def _record_loop(self):
        """Boucle d'enregistrement"""
        pa = pyaudio.PyAudio()
        
        try:
            self.stream = pa.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=pyaudio.paFloat32,
                input=True,
                frames_per_buffer=1024
            )
            
            while self.is_recording:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
                
        except Exception as e:
            logger.error(f"Erreur dans la boucle d'enregistrement: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            pa.terminate()


class AudioPipeline:
    """Pipeline audio principal intégrant STT, TTS et Wake Word Detection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        self.wake_word = WakeWordDetector(config)
        self.recorder = AudioRecorder()
        self.is_initialized = False
        self.mode = "realtime"  # ou "quality"
        
    def initialize(self, mode: str = "realtime"):
        """
        Initialise le pipeline audio
        
        Args:
            mode: Mode d'initialisation ("realtime" pour gaming, "quality" pour recherche)
        """
        logger.info("🎤 Initialisation du pipeline audio...")
        self.mode = mode
        
        try:
            # 1. Wake word detector (très léger)
            self.wake_word.load()
            
            # 2. STT - modèle adapté au mode
            if mode == "realtime":
                self.stt.load_light_model()  # Modèle léger pour gaming
            else:
                self.stt.load()  # Modèle complet pour qualité
            
            # 3. TTS
            self.tts.load()
            
            self.is_initialized = True
            logger.info("✅ Pipeline audio initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation pipeline: {e}")
            raise
    
    def switch_mode(self, new_mode: str):
        """
        Bascule entre les modes realtime et quality
        
        Args:
            new_mode: Nouveau mode ("realtime" ou "quality")
        """
        if new_mode == self.mode:
            return
        
        logger.info(f"🔄 Basculement vers mode {new_mode}...")
        
        # Décharge l'ancien modèle STT
        self.stt.unload()
        
        # Charge le nouveau modèle
        if new_mode == "realtime":
            self.stt.load_light_model()
        else:
            self.stt.load()
        
        self.mode = new_mode
        logger.info(f"✓ Mode {new_mode} activé")
    
    def start_listening(self, callback: Callable):
        """
        Démarre l'écoute du wake word
        
        Args:
            callback: Fonction appelée lors de la détection
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline non initialisé")
        
        self.wake_word.start_listening(callback)
    
    def stop_listening(self):
        """Arrête l'écoute"""
        self.wake_word.stop_listening()
    
    def record_and_transcribe(self, duration: float = None) -> str:
        """
        Enregistre de l'audio et le transcrit
        
        Args:
            duration: Durée d'enregistrement en secondes (None = manuel)
            
        Returns:
            Texte transcrit
        """
        # Démarrage de l'enregistrement
        self.recorder.start_recording()
        
        if duration:
            time.sleep(duration)
        else:
            input("Appuyez sur Entrée pour arrêter l'enregistrement...")
        
        # Arrêt et récupération de l'audio
        audio_data = self.recorder.stop_recording()
        
        # Transcription
        if len(audio_data) > 0:
            text = self.stt.transcribe(audio_data)
            return text
        else:
            return ""
    
    def speak(self, text: str, emotion: str = "neutral", save_path: str = None):
        """
        Génère et joue la réponse vocale
        
        Args:
            text: Texte à synthétiser
            emotion: Émotion de la voix
            save_path: Chemin pour sauvegarder l'audio (optionnel)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline non initialisé")
        
        # Génération de l'audio
        audio = self.tts.synthesize(text, emotion=emotion)
        
        if audio is not None:
            # Sauvegarde si demandé
            if save_path:
                self.tts.save(audio, save_path)
            
            # Lecture
            self.tts.play(audio)
    
    def cleanup(self):
        """Nettoie et libère les ressources"""
        logger.info("🧹 Nettoyage du pipeline audio...")
        
        self.stop_listening()
        self.stt.unload()
        self.tts.unload()
        self.wake_word.unload()
        
        self.is_initialized = False
        logger.info("✓ Pipeline audio nettoyé")


# Fonction utilitaire pour tester le pipeline
def test_pipeline():
    """Test basique du pipeline audio"""
    import json
    
    # Configuration de test
    config = {
        'tts': {
            'model': 'tts_models/multilingual/multi-dataset/your_tts',
            'sample_rate': 22050
        },
        'wake_word': {
            'keywords': ['computer'],
            'sensitivity': 0.5
        }
    }
    
    # Création du pipeline
    pipeline = AudioPipeline(config)
    
    try:
        # Initialisation
        pipeline.initialize(mode="realtime")
        
        # Test TTS
        print("\n🔊 Test de synthèse vocale...")
        pipeline.speak("Bonjour, je suis Kaguya, ton assistante vocale.")
        
        # Test STT
        print("\n🎤 Test de reconnaissance vocale (5 secondes)...")
        text = pipeline.record_and_transcribe(duration=5)
        print(f"Transcription: {text}")
        
        print("\n✅ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lancement des tests
    test_pipeline()
