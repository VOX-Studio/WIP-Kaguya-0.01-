# Architecture de Kaguya

## Vue d'ensemble

Kaguya est un assistant vocal autonome 100% local conçu pour fonctionner en parallèle du gaming sur Windows 11. L'architecture est modulaire et optimisée pour minimiser l'utilisation des ressources en mode Realtime.

## Modules principaux

### 1. Core (Orchestrateur)
**Fichier**: `core/agent.py`

Le cœur de Kaguya, responsable de:
- Orchestration des différents modules
- Gestion des états (idle, listening, thinking, speaking, etc.)
- Coordination des modes (Realtime, Qualité, Rest, Study)
- Planification et scheduling des tâches

**États de l'agent**:
- `IDLE`: En attente, comportements passifs
- `LISTENING`: Écoute active de l'utilisateur
- `THINKING`: Traitement et génération de réponse
- `SPEAKING`: Kaguya parle
- `RESEARCHING`: Recherche Wikipedia active
- `STUDYING`: Apprentissage autonome
- `RESTING`: Mode veille ultra-léger

### 2. Audio Pipeline
**Fichier**: `audio/pipeline.py`

Pipeline complet pour le traitement audio:

**Composants**:
- **VAD** (Voice Activity Detection): Détection de parole vs silence
- **Wake Word Detector**: Détection du mot-clé "Kaguya"
- **STT Engine**: Speech-to-Text avec Whisper local
- **TTS Engine**: Text-to-Speech avec voix réaliste et anime
- **Speaker Diarization**: Séparation et identification des locuteurs

**Flow audio**:
```
Microphone → VAD → Buffer → STT → Wake Word Check → Agent
                                                      ↓
Speaker ← TTS ← Response Generator ← ← ← ← ← ← ← ←  ←
```

### 3. Memory System
**Fichier**: `memory/memory_manager.py`

Système de mémoire à trois couches avec decay automatique:

**Trois types de mémoire**:
1. **Short-term** (court terme): Session actuelle, conversation en cours
2. **Long-term** (long terme): Préférences, décisions importantes, projets
3. **Knowledge**: Faits appris depuis Wikipedia, compressés

**Mécanismes**:
- **Priorités**: De TRIVIAL à CRITICAL
- **Decay**: Oubli progressif basé sur l'âge et les accès
- **Refresh**: Mise à jour tous les ~4 mois
- **Stockage**: SQLite pour persistance

### 4. Modes de fonctionnement

#### Mode Realtime (Low-Latency)
**Usage**: Pendant le gaming

**Caractéristiques**:
- Latence < 500ms
- Réponses courtes et rapides
- RAM limitée (~4 GB max)
- Pas de recherche longue
- VAD plus sensible

**Optimisations**:
- Modèles quantifiés si possible
- Batch size = 1
- Pas de recherche Wikipedia
- Cache de réponses communes

#### Mode Qualité (High-Quality)
**Usage**: Hors gaming, pour recherches

**Caractéristiques**:
- Recherche et synthèse Wikipedia
- Réponses détaillées
- RAM disponible (~16 GB)
- Progress updates affichés

**Capacités**:
- Recherche multi-sources
- Synthèse approfondie
- Création de knowledge entries
- Vérification de faits

#### Mode Rest (Veille)
**Usage**: Mise en veille

**Caractéristiques**:
- CPU < 5%
- RAM < 500 MB
- Détection wake word uniquement
- Réveil instantané sur "Kaguya"

#### Mode Study (Apprentissage)
**Usage**: Automatique pendant l'inactivité

**Caractéristiques**:
- Recherche Wikipedia autonome
- Alimentation knowledge base
- Checkpoints réguliers
- Interruptible à tout moment

### 5. Embodiment VTuber
**Fichier**: `embodiment/` (à implémenter)

Avatar VTuber sur 2e écran avec:

**États visuels**:
- Idle (repos)
- Listen (écoute)
- Think (réflexion)
- Speak (parole avec lip-sync)
- Pacing ("faire les 100 pas")
- Bored (s'ennuie)
- Sleeping (dort)

**Comportements**:
- Lip-sync sur la voix
- Gaze tracking (regarde l'utilisateur/écran)
- Mouvements procéduraux (pas de keyframes manuelles)
- Émotions simulées
- Peut bouder/ignorer temporairement

**Technologies suggérées**:
- VRM format pour le modèle 3D
- Unity ou Godot pour le rendering
- IPC avec Python pour le contrôle
- Alternative: Live2D pour 2D

### 6. Presence Detection
**Fichier**: `presence/` (à implémenter)

Détection de présence et attention:

**Webcam (optionnelle)**:
- Face detection
- Face recognition
- Attention estimation
- Présence physique

**Règles d'interaction**:
- Si webcam ON + pas de casque confirmé → pas de dialogue complet
- Peut interpeller brièvement pour attirer l'attention
- Confirmation casque par voix ou hotkey
- Interruptions intelligentes selon le contexte

### 7. Study Module
**Fichier**: `study/` (à implémenter)

Apprentissage autonome Wikipedia:

**Workflow**:
1. Trigger après 5min d'inactivité
2. Recherche d'articles pertinents
3. Extraction et compression
4. Création knowledge entries
5. Checkpoint régulier
6. Rapport sur demande

**Rendus**:
- "Qu'as-tu appris aujourd'hui ?"
- Résumé des interactions de la journée
- Statistiques d'apprentissage

### 8. Scheduler
**Fichier**: `scheduler/` (à implémenter)

Système de rappels et planification:

**Fonctionnalités**:
- "Rappelle-moi dans X minutes"
- "Rappelle-moi de Y demain"
- Vérification présence avant rappel
- Continuité du sujet

## Flow général d'interaction

```
1. User parle → Audio capturé
                    ↓
2. VAD détecte parole → Buffer audio
                            ↓
3. Fin de parole → STT transcription
                        ↓
4. Wake word check → Validation
                         ↓
5. Diarization → Identification speaker
                        ↓
6. Memory context retrieval
                ↓
7. Response generation (selon mode)
                ↓
8. Memory storage (conversation)
                ↓
9. TTS synthesis
                ↓
10. Audio playback + Embodiment animation
```

## Gestion des ressources

### Gaming Mode (Realtime)
```
GPU: ~2-3 GB VRAM
 - Whisper quantifié: 1-2 GB
 - TTS: 0.5-1 GB
 - Reste pour le jeu: 9-10 GB

RAM: ~4 GB
 - Application: 2 GB
 - Audio buffers: 0.5 GB
 - Memory cache: 1 GB
 - OS overhead: 0.5 GB

CPU: ~10-15%
 - VAD: 2-3%
 - Coordination: 5-8%
 - Embodiment: 5%
```

### Quality Mode (Hors gaming)
```
GPU: ~8 GB VRAM
 - Modèles full precision
 - Recherche et traitement

RAM: ~16 GB
 - Knowledge processing
 - Wikipedia cache
 - Recherche multi-sources

CPU: ~50-70%
 - Recherche web
 - Traitement NLP
 - Synthèse
```

## Configuration

Toute la configuration est centralisée dans `config.py`:

**Sections**:
- `HardwareConfig`: Spécifications matérielles
- `AudioConfig`: Paramètres audio STT/TTS
- `MemoryConfig`: Configuration mémoire
- `EmbodimentConfig`: Avatar VTuber
- `PresenceConfig`: Détection présence
- `StudyConfig`: Apprentissage autonome
- `InternetConfig`: Accès réseau (whitelist)
- `WakeConfig`: Wake word et rest mode

## Sécurité et confidentialité

**Principes**:
- 100% local, aucune donnée cloud
- Internet limité à Wikipedia par défaut
- Whitelist extensible avec consentement
- Données utilisateur chiffrées localement (optionnel)
- Pas de télémétrie

## Évolutivité

**Extensions futures possibles**:
- Support de plugins/skills
- Intégration smart home (Home Assistant)
- Multi-utilisateurs avec voiceprints
- Synchronisation cross-device (optionnel)
- API locale pour contrôle externe
- Support de langues additionnelles

## Limitations connues

**Version actuelle (base)**:
- TTS anime voice pas encore implémenté (placeholder)
- Embodiment VTuber non implémenté (interface externe recommandée)
- LLM local non intégré (responses placeholder)
- Speaker diarization simplifiée
- Pas de face recognition avancé

**Améliorations futures**:
- Intégrer un LLM local (LLaMA, Mistral)
- Améliorer la diarization avec pyannote.audio
- Implémenter RVC pour voix anime
- Créer l'embodiment complet
- Ajouter la détection d'émotion

## Technologies recommandées

**Core**:
- Python 3.10+
- PyTorch 2.1+ (avec CUDA 12.1)
- SQLite pour persistance

**Audio**:
- Whisper (openai/whisper-large-v3)
- Coqui TTS ou alternatives
- sounddevice pour capture/playback

**Computer Vision** (optionnel):
- OpenCV
- MediaPipe (face detection)

**Embodiment**:
- Unity avec VRM ou
- Godot avec VRM importer ou
- Live2D Cubism

**Monitoring**:
- psutil (CPU/RAM)
- GPUtil (GPU)
