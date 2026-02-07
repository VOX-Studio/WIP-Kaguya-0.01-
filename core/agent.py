"""
Core Agent - Orchestrateur principal de Kaguya
VERSION CORRIGÉE - last_interaction_time initialisé correctement
"""

import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import threading
import queue

from config import config, Mode, EmotionStyle, VoiceType
from memory.memory_manager import MemoryManager, MemoryEntry, MemoryType, MemoryPriority
from audio.pipeline import AudioPipeline


class AgentState(Enum):
    """États possibles de l'agent"""
    IDLE = "idle"              # En attente
    LISTENING = "listening"    # Écoute active
    THINKING = "thinking"      # Réflexion
    SPEAKING = "speaking"      # En train de parler
    RESEARCHING = "researching"  # Recherche Wikipedia
    STUDYING = "studying"      # Apprentissage autonome
    RESTING = "resting"        # Mode veille


@dataclass
class AgentContext:
    """Contexte actuel de l'agent"""
    current_mode: Mode = Mode.REALTIME
    current_state: AgentState = AgentState.IDLE
    current_emotion: EmotionStyle = EmotionStyle.NEUTRAL
    current_voice: VoiceType = VoiceType.ANIME
    
    user_present: bool = False
    user_attention: bool = False
    headset_confirmed: bool = False
    
    conversation_active: bool = False
    last_interaction_time: float = 0.0
    
    current_task: Optional[str] = None
    task_progress: float = 0.0


class ResponseGenerator:
    """Générateur de réponses (placeholder pour LLM)"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
    
    def generate(self, 
                 user_input: str,
                 context: AgentContext,
                 mode: Mode) -> str:
        """
        Générer une réponse
        
        Args:
            user_input: Input utilisateur
            context: Contexte actuel
            mode: Mode de fonctionnement
            
        Returns:
            Réponse générée
        """
        # Récupérer le contexte de mémoire
        memory_context = self.memory.get_context(max_entries=20)
        
        # TODO: Implémenter l'appel au LLM local
        # Pour l'instant, réponse simple
        
        if mode == Mode.REALTIME:
            # Réponse rapide et courte
            return self._generate_realtime_response(user_input, memory_context)
        elif mode == Mode.QUALITY:
            # Réponse détaillée avec recherche
            return self._generate_quality_response(user_input, memory_context)
        else:
            return "Je suis en mode veille. Dis 'Kaguya' pour me réveiller."
    
    def _generate_realtime_response(self, user_input: str, context: str) -> str:
        """Réponse mode Realtime (rapide)"""
        # Placeholder - à remplacer par LLM
        user_lower = user_input.lower()
        
        if "comment vas-tu" in user_lower or "ça va" in user_lower:
            return "Je vais bien, merci ! Et toi ?"
        elif "quoi de neuf" in user_lower or "nouveau" in user_lower:
            return "Rien de spécial pour l'instant. Tu veux discuter de quelque chose ?"
        elif "merci" in user_lower:
            return "De rien, c'est un plaisir !"
        elif "bye" in user_lower or "à plus" in user_lower:
            return "À bientôt ! N'hésite pas si tu as besoin de moi."
        else:
            return "Je t'écoute, que veux-tu savoir ?"
    
    def _generate_quality_response(self, user_input: str, context: str) -> str:
        """Réponse mode Qualité (avec recherche)"""
        # Placeholder - à implémenter avec recherche Wikipedia
        return f"Je vais chercher des informations sur '{user_input}'. Un instant..."


class KaguayAgent:
    """Agent principal de Kaguya"""
    
    def __init__(self):
        """Initialiser l'agent"""
        self.config = config
        self.context = AgentContext()
        
        # Composants
        self.memory = MemoryManager(self.config.memory.memory_dir)
        self.audio = AudioPipeline(
            sample_rate=self.config.audio.sample_rate,
            chunk_size=self.config.audio.chunk_size,
            channels=self.config.audio.channels
        )
        self.response_generator = ResponseGenerator(self.memory)
        
        # État
        self.running = False
        self.interaction_queue = queue.Queue()
        
        # Threads
        self.main_thread: Optional[threading.Thread] = None
        self.study_thread: Optional[threading.Thread] = None
        
        print("🌸 Kaguya initialisée")
    
    def start(self):
        """Démarrer l'agent"""
        print("\n" + "="*50)
        print("🚀 Démarrage de Kaguya")
        print("="*50)
        
        # Initialiser les composants
        self.audio.initialize()
        
        # Charger la mémoire long terme
        self._load_persistent_memory()
        
        # Démarrer le mode par défaut
        self.switch_mode(self.config.default_mode)
        
        # Démarrer la boucle principale
        self.running = True
        
        # ✅ CORRECTION CRITIQUE: Initialiser last_interaction_time au démarrage
        # Sinon il reste à 0 et idle_time devient énorme → recherches immédiates!
        self.context.last_interaction_time = time.time()
        
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.start()
        
        # Démarrer l'écoute
        self.audio.start_listening(self._on_user_input)
        
        print("✓ Kaguya est prête !\n")
    
    def stop(self):
        """Arrêter l'agent"""
        print("\n🛑 Arrêt de Kaguya...")
        
        self.running = False
        self.audio.stop_listening()
        
        if self.main_thread:
            self.main_thread.join()
        
        # Sauvegarder la mémoire
        self._save_persistent_memory()
        
        print("✓ Kaguya arrêtée proprement")
    
    def switch_mode(self, mode: Mode):
        """Changer de mode de fonctionnement"""
        print(f"🔄 Passage en mode {mode.value}")
        self.context.current_mode = mode
        
        # Ajuster les ressources selon le mode
        if mode == Mode.REALTIME:
            # Optimiser pour faible latence
            self.audio.vad.threshold = 0.02  # ✅ CORRIGÉ: 0.02 au lieu de 0.4
        elif mode == Mode.QUALITY:
            # Privilégier la qualité
            self.audio.vad.threshold = 0.03  # ✅ CORRIGÉ: 0.03 au lieu de 0.6
        elif mode == Mode.REST:
            # Mode veille ultra léger
            self.context.current_state = AgentState.RESTING
    
    def _main_loop(self):
        """Boucle principale de l'agent"""
        while self.running:
            try:
                # Vérifier s'il y a des interactions en attente
                try:
                    interaction = self.interaction_queue.get(timeout=0.1)
                    self._process_interaction(interaction)
                except queue.Empty:
                    pass
                
                # Vérifier si on doit passer en mode Study
                self._check_study_trigger()
                
                # Appliquer le decay de mémoire périodiquement
                if time.time() % 3600 < 1:  # Toutes les heures
                    self._apply_memory_decay()
                
                # Comportements idle
                if self.context.current_state == AgentState.IDLE:
                    self._idle_behaviors()
                
                time.sleep(0.1)
            
            except Exception as e:
                print(f"Erreur dans main loop: {e}")
    
    def _on_user_input(self, transcription: str):
        """Callback quand l'utilisateur parle"""
        print(f"\n👤 Utilisateur: {transcription}")
        
        # Mettre à jour le contexte
        self.context.last_interaction_time = time.time()
        self.context.conversation_active = True
        
        # Ajouter à la queue d'interaction
        self.interaction_queue.put({
            'type': 'user_input',
            'content': transcription,
            'timestamp': time.time()
        })
    
    def _process_interaction(self, interaction: Dict[str, Any]):
        """Traiter une interaction utilisateur"""
        if interaction['type'] != 'user_input':
            return
        
        user_input = interaction['content']
        
        # Sauvegarder dans la mémoire court terme
        self.memory.add(MemoryEntry(
            content=f"User: {user_input}",
            memory_type=MemoryType.SHORT_TERM.value,
            priority=MemoryPriority.MEDIUM.value,
            tags=["conversation"]
        ))
        
        # Changer l'état
        self.context.current_state = AgentState.THINKING
        
        # Générer la réponse
        response = self.response_generator.generate(
            user_input,
            self.context,
            self.context.current_mode
        )
        
        # Sauvegarder la réponse
        self.memory.add(MemoryEntry(
            content=f"Kaguya: {response}",
            memory_type=MemoryType.SHORT_TERM.value,
            priority=MemoryPriority.MEDIUM.value,
            tags=["conversation"]
        ))
        
        # Parler
        self.context.current_state = AgentState.SPEAKING
        self.audio.speak(
            response,
            emotion=self.context.current_emotion.value,
            voice=self.context.current_voice.value
        )
        
        # Retour à idle
        self.context.current_state = AgentState.IDLE
    
    def _check_study_trigger(self):
        """Vérifier si on doit démarrer le mode Study"""
        if not self.config.study.enable_autonomous_study:
            return
        
        if self.context.current_state != AgentState.IDLE:
            return
        
        # Vérifier le temps d'inactivité
        idle_time = time.time() - self.context.last_interaction_time
        
        # ✅ Debug ajouté pour surveiller
        if idle_time > 60 and idle_time % 60 < 0.2:  # Afficher chaque minute
            print(f"⏱️  Inactivité: {idle_time:.0f}s / {self.config.study.study_idle_time_threshold_s:.0f}s")
        
        if idle_time > self.config.study.study_idle_time_threshold_s:
            if self.study_thread is None or not self.study_thread.is_alive():
                print("\n📚 Démarrage du mode Study...")
                self.study_thread = threading.Thread(target=self._study_loop)
                self.study_thread.start()
    
    def _study_loop(self):
        """Boucle d'apprentissage autonome"""
        self.context.current_state = AgentState.STUDYING
        
        # TODO: Implémenter la recherche et synthèse Wikipedia
        print("📖 Apprentissage en cours...")
        
        # Placeholder
        time.sleep(10)
        
        print("✓ Session d'étude terminée")
        self.context.current_state = AgentState.IDLE
    
    def _idle_behaviors(self):
        """Comportements quand idle"""
        # TODO: Implémenter les comportements idle pour l'avatar
        # (pacing, bored, etc.)
        pass
    
    def _apply_memory_decay(self):
        """Appliquer le decay sur toutes les mémoires"""
        for memory_type in MemoryType:
            deleted, updated = self.memory.apply_decay(
                memory_type,
                self.config.memory.decay_factor_days,
                self.config.memory.min_priority_threshold
            )
            if deleted > 0 or updated > 0:
                print(f"🧹 Memory cleanup [{memory_type.value}]: "
                      f"{deleted} deleted, {updated} updated")
    
    def _load_persistent_memory(self):
        """Charger la mémoire persistante au démarrage"""
        stats = self.memory.stats()
        print(f"💾 Mémoire chargée: {stats}")
    
    def _save_persistent_memory(self):
        """Sauvegarder la mémoire persistante"""
        # La mémoire est déjà en SQLite, pas besoin de sauvegarde explicite
        stats = self.memory.stats()
        print(f"💾 Mémoire sauvegardée: {stats}")


if __name__ == "__main__":
    # Test de l'agent
    agent = KaguayAgent()
    
    try:
        agent.start()
        
        # Simuler une conversation
        print("\n" + "="*50)
        print("Agent en cours d'exécution. Appuyez sur Ctrl+C pour arrêter.")
        print("="*50 + "\n")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nInterruption détectée...")
    finally:
        agent.stop()