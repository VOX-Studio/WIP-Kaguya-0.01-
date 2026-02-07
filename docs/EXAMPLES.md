# Exemples d'utilisation de Kaguya

## 🎯 Scénarios d'usage

### Scénario 1: Gaming avec assistance vocale

```bash
# Lancer en mode Realtime optimisé pour le gaming
python main.py --mode realtime
```

**Conversation exemple**:
```
Toi: "Kaguya"
Kaguya: *se réveille* "Oui ?"

Toi: "Rappelle-moi de check les objectifs quotidiens dans 30 minutes"
Kaguya: "D'accord, je te rappellerai dans 30 minutes."

Toi: "Quelle est la meilleure build pour un mage de feu ?"
Kaguya: "Pour un mage de feu, privilégie l'INT et les équipements qui boostent les dégâts de feu..."
```

### Scénario 2: Recherche approfondie

```bash
# Lancer en mode Qualité pour recherches
python main.py --mode quality
```

**Conversation exemple**:
```
Toi: "Kaguya, parle-moi de l'histoire de l'intelligence artificielle"
Kaguya: "Je vais chercher des informations sur ce sujet. Un instant..."
        *[Progress: Recherche Wikipedia...]*
        *[Progress: Analyse de 5 articles...]*
        *[Progress: Synthèse des informations...]*
        "L'intelligence artificielle a des origines qui remontent aux années 1950..."
        *[Réponse détaillée de 2-3 minutes]*

Toi: "Sauvegarde ça dans ta mémoire"
Kaguya: "C'est noté. Je m'en souviendrai."
```

### Scénario 3: Apprentissage autonome

```python
# Dans le code
agent = KaguayAgent()
agent.start()

# Kaguya va automatiquement passer en mode Study après 5 minutes d'inactivité
# Elle va apprendre depuis Wikipedia et alimenter sa knowledge base
```

**Ce que Kaguya apprend**:
- Sujets liés aux conversations récentes
- Domaines d'intérêt de l'utilisateur
- Faits généraux pour enrichir ses réponses

**Après une session d'apprentissage**:
```
Toi: "Kaguya, qu'as-tu appris aujourd'hui ?"
Kaguya: "Aujourd'hui, j'ai étudié 8 articles sur les jeux vidéo RPG, 
         l'histoire du gaming, et quelques concepts de game design. 
         J'ai aussi exploré des articles sur la mythologie japonaise 
         puisque tu en as parlé ce matin."
```

## 🔧 Exemples de code

### Utiliser la mémoire

```python
from memory import MemoryManager, MemoryEntry, MemoryType, MemoryPriority

# Créer le manager
memory = MemoryManager("./data/memory")

# Ajouter une préférence utilisateur
entry = MemoryEntry(
    content="L'utilisateur préfère les réponses concises",
    memory_type=MemoryType.LONG_TERM.value,
    priority=MemoryPriority.HIGH.value,
    tags=["preferences", "communication"]
)
memory.add(entry)

# Rechercher dans la mémoire
results = memory.search(
    query="préférences",
    memory_type=MemoryType.LONG_TERM,
    min_priority=0.5
)

for result in results:
    print(f"- {result.content}")

# Obtenir le contexte pour l'agent
context = memory.get_context(max_entries=20)
print(context)
```

### Utiliser l'audio pipeline

```python
from audio.pipeline import AudioPipeline

# Créer le pipeline
pipeline = AudioPipeline(
    sample_rate=16000,
    chunk_size=1024
)

# Initialiser
pipeline.initialize()

# Définir le callback
def on_transcription(text):
    print(f"User said: {text}")
    # Générer une réponse
    response = "Tu as dit: " + text
    # Faire parler Kaguya
    pipeline.speak(response, emotion="joyeux")

# Démarrer l'écoute
pipeline.start_listening(on_transcription)

# ... laisser tourner ...

# Arrêter
pipeline.stop_listening()
```

### Créer un agent personnalisé

```python
from core.agent import KaguayAgent
from config import config, Mode, EmotionStyle

# Personnaliser la config
config.user_name = "Senpai"
config.default_voice = VoiceType.ANIME
config.wake.wake_word = "hey_assistant"

# Créer l'agent
agent = KaguayAgent()

# Démarrer
agent.start()

# Changer de mode à la volée
agent.switch_mode(Mode.QUALITY)

# ... utiliser l'agent ...

# Arrêter
agent.stop()
```

### Système d'émotions

```python
# Dans une réponse
if user_said_thanks:
    agent.context.current_emotion = EmotionStyle.JOYEUX
    agent.audio.speak("De rien, c'est un plaisir !", emotion="joyeux")
elif user_seems_frustrated:
    agent.context.current_emotion = EmotionStyle.DECU
    agent.audio.speak("Je suis désolée si je n'ai pas été utile...", emotion="triste")
```

## 🎨 Personnalisation

### Changer la personnalité

```python
# Dans config.py ou le JSON
config.user_name = "Maître"  # Comment Kaguya t'appelle

# Créer un prompt system personnalisé (à venir dans LLM integration)
KAGUYA_PERSONALITY = """
Tu es Kaguya, une assistante vocale anime avec une personnalité tsundere.
Tu es intelligente et compétente, mais tu as parfois tendance à être un peu
moqueuse avec ton maître. Tu l'apprécies vraiment mais tu ne veux pas 
toujours le montrer directement.
"""
```

### Créer des réponses personnalisées

```python
from core.agent import ResponseGenerator

class CustomResponseGenerator(ResponseGenerator):
    def _generate_realtime_response(self, user_input, context):
        user_lower = user_input.lower()
        
        # Réponses personnalisées
        if "tu es la meilleure" in user_lower:
            return "Hmph, évidemment que je suis la meilleure ! Mais... merci."
        elif "je t'aime" in user_lower:
            return "Baka ! Ne dis pas des choses aussi embarrassantes !"
        
        # Fallback
        return super()._generate_realtime_response(user_input, context)
```

### Ajouter des comportements idle

```python
# Dans core/agent.py

def _idle_behaviors(self):
    """Comportements quand idle"""
    idle_time = time.time() - self.context.last_interaction_time
    
    # Après 5 minutes
    if idle_time > 300:
        if random.random() < 0.1:  # 10% de chance
            self.audio.speak("Je m'ennuie un peu...", emotion="agace")
    
    # Après 15 minutes
    if idle_time > 900:
        if random.random() < 0.05:  # 5% de chance
            self.audio.speak("*bâillement* Tu fais quoi ?", emotion="bored")
```

## 🎮 Intégration avec des jeux

### Overlay gaming (concept)

```python
# Utiliser PyAutoGUI ou similaire pour overlay
import pyautogui
from PIL import Image, ImageDraw, ImageFont

def create_overlay(text, emotion="neutral"):
    """Créer un overlay transparent pour Kaguya"""
    img = Image.new('RGBA', (400, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Bulle de dialogue
    draw.rectangle([(10, 10), (390, 90)], fill=(0, 0, 0, 180), outline=(255, 255, 255))
    
    # Texte
    font = ImageFont.truetype("arial.ttf", 16)
    draw.text((20, 20), f"🌸 Kaguya: {text}", font=font, fill=(255, 255, 255))
    
    # Afficher temporairement
    # ... (code pour afficher en overlay)
```

### Commandes vocales gaming

```python
GAMING_COMMANDS = {
    "sauvegarde": lambda: keyboard.press('F5'),
    "inventaire": lambda: keyboard.press('i'),
    "carte": lambda: keyboard.press('m'),
    "pause": lambda: keyboard.press('esc'),
}

def on_gaming_command(transcription):
    text_lower = transcription.lower()
    
    for command, action in GAMING_COMMANDS.items():
        if command in text_lower:
            action()
            agent.audio.speak(f"Commande {command} exécutée", emotion="neutral")
            break
```

## 📊 Monitoring et debug

### Afficher les stats en temps réel

```python
import psutil
import GPUtil

def show_stats():
    """Afficher les stats système"""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # RAM
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    
    # GPU
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_usage = gpu.load * 100
        vram_used = gpu.memoryUsed
    
    print(f"""
    📊 System Stats:
    CPU: {cpu_percent}%
    RAM: {ram_used_gb:.1f} GB / {ram.total / (1024**3):.1f} GB
    GPU: {gpu_usage:.1f}%
    VRAM: {vram_used} MB
    """)

# Appeler périodiquement
import threading

def stats_loop():
    while running:
        show_stats()
        time.sleep(10)

stats_thread = threading.Thread(target=stats_loop)
stats_thread.start()
```

### Logger les conversations

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'logs/conversation_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_interaction(user_input, kaguya_response):
    logging.info(f"USER: {user_input}")
    logging.info(f"KAGUYA: {kaguya_response}")
```

## 🚀 Cas d'usage avancés

### Multi-utilisateur

```python
# Enrôler plusieurs utilisateurs
agent.audio.diarization.enroll_speaker("User1", audio_sample_1)
agent.audio.diarization.enroll_speaker("User2", audio_sample_2)

# Identifier qui parle
speaker = agent.audio.diarization.identify_speaker(audio)

# Adapter les réponses
if speaker == "User1":
    response = "Bonjour Maître !"
elif speaker == "User2":
    response = "Ah, c'est toi. Bonjour."
```

### Intégration smart home (concept)

```python
# Via Home Assistant API ou similaire
def control_lights(action, room):
    if action == "allumer":
        # API call to turn on lights
        return f"J'allume les lumières de {room}"
    elif action == "éteindre":
        # API call to turn off lights
        return f"J'éteins les lumières de {room}"

# Dans le response generator
if "allume" in user_input and "lumière" in user_input:
    response = control_lights("allumer", "salon")
```

## 📝 Templates de configuration

### Configuration gaming optimale

```json
{
  "default_mode": "realtime",
  "hardware": {
    "max_ram_gaming_mode_gb": 4.0,
    "max_vram_usage_gb": 3.0
  },
  "audio": {
    "vad_threshold": 0.4,
    "stt_model": "openai/whisper-medium"
  },
  "embodiment": {
    "fps_target": 30
  }
}
```

### Configuration recherche/productivité

```json
{
  "default_mode": "quality",
  "hardware": {
    "max_ram_quality_mode_gb": 16.0,
    "max_vram_usage_gb": 10.0
  },
  "study": {
    "enable_autonomous_study": true,
    "max_articles_per_session": 20
  }
}
```

---

Ces exemples montrent le potentiel de Kaguya. Beaucoup nécessitent l'implémentation complète des TODOs, mais la structure est là pour les supporter ! 🌸
