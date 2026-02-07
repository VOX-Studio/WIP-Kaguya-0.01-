"""
Pipeline audio de Kaguya: STT, TTS, VAD, Diarization
VERSION CORRIGÉE - Seuil VAD ajusté + TTS fonctionnel
"""

import numpy as np
import sounddevice as sd
import queue
import threading
import os
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
import wave
import io
import time


@dataclass
class AudioChunk:
    """Chunk audio avec métadonnées"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speech: bool = False
    speaker_id: Optional[str] = None


class VAD:
    """Voice Activity Detection"""
    
    def __init__(self, 
                 threshold: float = 0.02,  # ✅ CORRIGÉ: 0.02 au lieu de 0.5
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 500):
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_duration_ms * 16)  # 16kHz
        self.min_silence_samples = int(min_silence_duration_ms * 16)
        
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
    
    def __call__(self, audio_chunk: np.ndarray) -> bool:
        """
        Détecter si le chunk contient de la parole
        
        Args:
            audio_chunk: Audio data (numpy array)
            
        Returns:
            True si parole détectée, False sinon
        """
        # ✅ CORRECTION: Gérer le stéréo -> mono
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # Calcul RMS (Root Mean Square) comme proxy simple
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        is_speech = rms > self.threshold
        
        if is_speech:
            self.speech_frames += len(audio_chunk)
            self.silence_frames = 0
            
            if self.speech_frames >= self.min_speech_samples:
                self.is_speaking = True
        else:
            self.silence_frames += len(audio_chunk)
            self.speech_frames = 0
            
            if self.silence_frames >= self.min_silence_samples:
                self.is_speaking = False
        
        return self.is_speaking


class WakeWordDetector:
    """Détection du wake word (mot-clé de réveil)"""
    
    def __init__(self, wake_word: str = "kaguya", confidence_threshold: float = 0.8):
        self.wake_word = wake_word.lower()
        self.confidence_threshold = confidence_threshold
        self.enabled = False  # ✅ DÉSACTIVÉ par défaut - accepte toute parole
    
    def detect(self, transcription: str) -> bool:
        """
        Vérifier si le wake word est présent
        
        Args:
            transcription: Texte transcrit
            
        Returns:
            True si wake word détecté
        """
        if not self.enabled:
            return True  # Toujours accepter si désactivé
        
        text_lower = transcription.lower()
        
        # Simple substring match (à améliorer avec phonetic matching)
        is_detected = self.wake_word in text_lower
        
        if is_detected:
            print(f"✓ Wake word '{self.wake_word}' détecté!")
        
        return is_detected


class STTEngine:
    """Speech-to-Text Engine (Whisper local)"""
    
    def __init__(self, 
                 model_name: str = "openai/whisper-large-v3",
                 device: str = "cuda",
                 language: str = "fr"):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self.processor = None
    
    def load(self):
        """Charger le modèle Whisper"""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch
            
            self.processor = cache_dir = "./models_cache"
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                local_files_only=True,  # IMPORTANT
                token=os.getenv("HF_TOKEN", None),
)
            
            # ✅ CORRECTION: Utiliser dtype au lieu de torch_dtype (déprécié)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=dtype,
                low_cpu_mem_usage=True  # Optimisation mémoire
            ).to(self.device)
            
            print(f"✓ Modèle STT chargé: {self.model_name} (dtype={dtype})")
        except Exception as e:
            print(f"✗ Erreur chargement STT: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcrire l'audio en texte
        
        Args:
            audio: Audio data (numpy array)
            sample_rate: Sample rate
            
        Returns:
            Texte transcrit
        """
        if self.model is None:
            raise RuntimeError("Modèle STT non chargé")
        
        try:
            import torch
            
            # ✅ CORRECTION: S'assurer que l'audio est en float32 pour le processor
            audio = audio.astype(np.float32)
            
            # Prétraitement
            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            # ✅ CORRECTION: Convertir au bon dtype selon le device
            if self.device == "cuda":
                inputs = inputs.to(self.device, dtype=torch.float16)
            else:
                inputs = inputs.to(self.device)
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    language=self.language,
                    max_length=448,
                    # ✅ Éviter le warning attention_mask
                    return_timestamps=False,
                    task="transcribe"
                )
            
            # Décodage
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
        
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""


class TTSEngine:
    """Text-to-Speech Engine avec support multi-voix"""
    
    def __init__(self,
                 voice_type: str = "gtts",  # "gtts", "coqui", "bark"
                 device: str = "cuda",
                 language: str = "fr"):
        self.voice_type = voice_type
        self.device = device
        self.language = language
        self.model = None
        self.sample_rate = 22050
    
    def load(self):
        """Charger le moteur TTS"""
        print(f"🔊 Chargement TTS: {self.voice_type}")
        
        if self.voice_type == "coqui":
            try:
                from TTS.api import TTS
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                print("✓ Coqui XTTS-v2 chargé")
            except Exception as e:
                print(f"⚠️  Coqui non disponible ({e}), utilisation de gTTS")
                self.voice_type = "gtts"
        
        elif self.voice_type == "bark":
            try:
                from transformers import AutoProcessor, BarkModel
                import torch
                self.processor = AutoProcessor.from_pretrained("suno/bark-small")
                self.model = BarkModel.from_pretrained(
                    "suno/bark-small",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                print("✓ Bark TTS chargé")
            except Exception as e:
                print(f"⚠️  Bark non disponible ({e}), utilisation de gTTS")
                self.voice_type = "gtts"
        
        else:  # gtts par défaut
            try:
                from gtts import gTTS
                print("✓ gTTS prêt (nécessite Internet)")
            except ImportError:
                print("❌ gTTS non installé! pip install gtts pydub")
                raise
    
    def synthesize(self, 
                   text: str,
                   emotion: str = "neutral",
                   speed: float = 1.0) -> np.ndarray:
        """
        Synthétiser la parole
        
        Args:
            text: Texte à prononcer
            emotion: Émotion (neutral, joyeux, etc.)
            speed: Vitesse de parole
            
        Returns:
            Audio en numpy array (float32, 22050Hz)
        """
        print(f"🎙️  TTS: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        
        try:
            if self.voice_type == "coqui":
                return self._synthesize_coqui(text, emotion)
            elif self.voice_type == "bark":
                return self._synthesize_bark(text, emotion)
            else:
                return self._synthesize_gtts(text, speed)
        except Exception as e:
            print(f"❌ Erreur TTS: {e}")
            # Fallback: retourner un court silence plutôt qu'un bip
            return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
    
    def _synthesize_coqui(self, text: str, emotion: str) -> np.ndarray:
        """Synthèse avec Coqui XTTS-v2"""
        # Voix de référence optionnelle (pour clonage de voix anime)
        speaker_wav = None  # Mettre "./assets/voices/reference_anime.wav" si disponible
        
        if speaker_wav and os.path.exists(speaker_wav):
            audio = self.model.tts(text=text, language=self.language, speaker_wav=speaker_wav)
        else:
            audio = self.model.tts(text=text, language=self.language)
        
        return np.array(audio, dtype=np.float32)
    
    def _synthesize_bark(self, text: str, emotion: str) -> np.ndarray:
        """Synthèse avec Bark"""
        import torch
        
        # Ajouter des marqueurs émotionnels
        if emotion == "joyeux":
            text = f"[laughs] {text}"
        elif emotion == "excite":
            text = f"!! {text} !!"
        elif emotion == "triste":
            text = f"... {text}"
        
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            audio = self.model.generate(**inputs)
        
        audio_np = audio.cpu().numpy().squeeze()
        
        # Bark génère à 24kHz, resample à 22050Hz
        if len(audio_np) > 0:
            from scipy.signal import resample
            target_length = int(len(audio_np) * self.sample_rate / 24000)
            audio_np = resample(audio_np, target_length)
        
        return audio_np.astype(np.float32)
    
    def _synthesize_gtts(self, text: str, speed: float) -> np.ndarray:
        """Synthèse avec Google TTS (simple et fiable)"""
        from gtts import gTTS
        import tempfile
        
        # Générer l'audio
        tts = gTTS(text=text, lang='fr', slow=(speed < 0.8))
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tts.save(tmp.name)
            
            # Lire le MP3
            try:
                # Option 1: avec soundfile (nécessite ffmpeg)
                import soundfile as sf
                audio, sr = sf.read(tmp.name)
            except:
                # Option 2: avec pydub (plus robuste)
                from pydub import AudioSegment
                sound = AudioSegment.from_mp3(tmp.name)
                
                # Convertir en numpy
                audio = np.array(sound.get_array_of_samples(), dtype=np.float32)
                audio = audio / 32768.0  # Normaliser de int16 à float32
                sr = sound.frame_rate
                
                # Convertir stéréo en mono si nécessaire
                if sound.channels == 2:
                    audio = audio.reshape((-1, 2)).mean(axis=1)
            
            # Resample si nécessaire
            if sr != self.sample_rate:
                from scipy.signal import resample
                target_length = int(len(audio) * self.sample_rate / sr)
                audio = resample(audio, target_length)
            
            # Cleanup
            os.unlink(tmp.name)
            
            return audio.astype(np.float32)


class SpeakerDiarization:
    """Diarization et reconnaissance des locuteurs"""
    
    def __init__(self):
        self.voiceprints: Dict[str, np.ndarray] = {}  # speaker_id -> embedding
        self.current_speaker: Optional[str] = None
    
    def enroll_speaker(self, speaker_id: str, audio: np.ndarray):
        """
        Enrôler un nouveau locuteur
        
        Args:
            speaker_id: Identifiant du locuteur
            audio: Échantillon audio pour le voiceprint
        """
        # TODO: Implémenter l'extraction d'embedding vocal
        # Pour l'instant, placeholder
        embedding = np.random.randn(128)  # Placeholder
        self.voiceprints[speaker_id] = embedding
        print(f"✓ Locuteur '{speaker_id}' enrôlé")
    
    def identify_speaker(self, audio: np.ndarray, threshold: float = 0.75) -> Optional[str]:
        """
        Identifier le locuteur depuis l'audio
        
        Args:
            audio: Audio data
            threshold: Seuil de confiance
            
        Returns:
            speaker_id si identifié, None sinon
        """
        if not self.voiceprints:
            return None
        
        # TODO: Implémenter la reconnaissance réelle
        # Pour l'instant, retourne le premier speaker
        return list(self.voiceprints.keys())[0] if self.voiceprints else None


class AudioPipeline:
    """Pipeline audio principal"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        # Composants - ✅ CORRIGÉ: seuil VAD ajusté
        self.vad = VAD(threshold=0.02)  # Au lieu de VAD() avec seuil par défaut
        self.wake_detector = WakeWordDetector()
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.diarization = SpeakerDiarization()
        
        # État
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.recording_buffer = []
        self.transcription_callback: Optional[Callable] = None
        
        # Thread d'enregistrement
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def initialize(self):
        """Initialiser le pipeline"""
        print("Initialisation du pipeline audio...")
        self.stt.load()
        self.tts.load()
        print("✓ Pipeline audio initialisé")
    
    def start_listening(self, callback: Callable[[str], None]):
        """
        Démarrer l'écoute continue
        
        Args:
            callback: Fonction appelée avec la transcription
        """
        self.transcription_callback = callback
        self.is_listening = True
        self.stop_event.clear()
        
        # Démarrer le thread d'enregistrement
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.start()
        
        print("🎤 Écoute démarrée")
    
    def stop_listening(self):
        """Arrêter l'écoute"""
        self.is_listening = False
        self.stop_event.set()
        
        if self.recording_thread:
            self.recording_thread.join()
        
        print("🔇 Écoute arrêtée")
    
    def _recording_loop(self):
        """Boucle d'enregistrement audio"""
        def audio_callback(indata, frames, time_info, status):
            """Callback pour sounddevice"""
            if status:
                print(f"Status: {status}")
            self.audio_queue.put(indata.copy())
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            blocksize=self.chunk_size,
            latency='low'  # ✅ Réduire la latence
        ):
            while not self.stop_event.is_set():
                try:
                    # Récupérer chunk audio
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # VAD
                    is_speech = self.vad(audio_chunk)
                    
                    if is_speech:
                        self.recording_buffer.append(audio_chunk)
                        # ✅ Afficher seulement au début de la parole
                        if len(self.recording_buffer) == 1:
                            print("🗣️ Début de parole détectée...")
                    elif len(self.recording_buffer) > 0:
                        # Fin de parole détectée, transcrire
                        print(f"✅ Fin de parole détectée, {len(self.recording_buffer)} chunks à transcrire")
                        self._process_recording()
                        self.recording_buffer = []
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Erreur dans recording loop: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _process_recording(self):
        """Traiter l'enregistrement audio"""
        if not self.recording_buffer:
            return
        
        try:
            # Concaténer les chunks
            audio = np.concatenate(self.recording_buffer)
            
            # ✅ CORRECTION: Convertir en mono si nécessaire
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Moyenne des canaux
            
            # ✅ CORRECTION: S'assurer que l'audio est float32
            audio = audio.astype(np.float32)
            
            # ✅ CORRECTION: Normaliser l'audio entre -1 et 1
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            print(f"🎵 Audio préparé: shape={audio.shape}, dtype={audio.dtype}, range=[{audio.min():.3f}, {audio.max():.3f}]")
            
            # Transcrire
            transcription = self.stt.transcribe(audio, self.sample_rate)
            
            if not transcription:
                print("⚠️ Transcription vide")
                return
            
            print(f"📝 Transcription: {transcription}")
            
            # Vérifier wake word (désactivé par défaut)
            if not self.wake_detector.detect(transcription):
                print("⏸️  Wake word non détecté, parole ignorée")
                return
            
            # Si wake word désactivé, on continue directement
            # Identifier le locuteur
            speaker_id = self.diarization.identify_speaker(audio)
            
            # Callback
            if self.transcription_callback:
                self.transcription_callback(transcription)
        
        except Exception as e:
            print(f"❌ Erreur dans _process_recording: {e}")
            import traceback
            traceback.print_exc()
    
    def speak(self, text: str, emotion: str = "neutral", voice: str = "realistic"):
        """
        Faire parler Kaguya
        
        Args:
            text: Texte à prononcer
            emotion: Style émotionnel
            voice: Type de voix
        """
        print(f"🗣️ Kaguya: {text}")
        
        # Synthétiser
        audio = self.tts.synthesize(text, emotion)
        
        # Jouer
        sd.play(audio, samplerate=22050)
        sd.wait()


if __name__ == "__main__":
    # Test du pipeline
    pipeline = AudioPipeline()
    pipeline.initialize()
    
    def on_transcription(text: str):
        print(f"Reçu: {text}")
        pipeline.speak(f"Tu as dit: {text}")
    
    # Test vocal (décommenter pour tester)
    # pipeline.start_listening(on_transcription)
    # time.sleep(30)  # Écouter pendant 30 secondes
    # pipeline.stop_listening()
    
    # Test TTS
    pipeline.speak("Bonjour, je suis Kaguya !", emotion="joyeux")