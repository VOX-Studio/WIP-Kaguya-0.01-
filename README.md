# Kaguya - Assistant Vocal Autonome Local

Agent vocal autonome 100% local sur Windows 11 avec embodiment VTuber, optimisé pour gaming sur RTX 4070 Super.

## 🎯 Caractéristiques principales

- **Dual-Mode**: Realtime (low-latency gaming) + Qualité (recherche & synthèse)
- **Embodiment VTuber**: Avatar animé sur 2e écran avec émotions et comportements vivants
- **Mémoire solide**: Court/long terme + knowledge base Wikipédia
- **Voice Intelligence**: STT/TTS, diarization, voiceprint recognition
- **Safe Autonomy**: Internet limité à Wikipédia par défaut
- **Resource-Aware**: RAM minimale en mode gaming

## 🏗️ Architecture

```
kaguya/
├── core/               # Orchestrateur principal et logique d'agent
├── audio/              # Pipeline STT/TTS et traitement audio
├── memory/             # Système de mémoire (court/long terme/knowledge)
├── embodiment/         # Moteur VTuber et animations procédurales
├── modes/              # Modes Realtime et Qualité
├── voice/              # Gestion des voix et émotions
├── presence/           # Détection présence et attention (webcam opt.)
├── study/              # Mode apprentissage autonome Wikipédia
├── scheduler/          # Rappels et planification
├── ui/                 # Interface utilisateur Windows
└── config/             # Configurations et paramètres
```

## 🚀 Démarrage rapide

```bash
# Installation des dépendances
pip install -r requirements.txt

# Configuration initiale
python setup.py

# Lancement
python main.py
```

## 📋 Prérequis

- Windows 11
- Python 3.10+
- CUDA 12.x (pour RTX 4070 Super)
- 32 GB RAM
- Écran secondaire (pour embodiment)

## 🔧 Configuration matérielle

- **GPU**: RTX 4070 Super (12GB VRAM)
- **CPU**: i7-13700KF
- **RAM**: 32 GB DDR4/DDR5
- **Optimisation**: Priorité RAM minimale en mode gaming

## 📖 Documentation

Voir `/docs` pour la documentation complète de chaque module.

## 🎮 Modes d'utilisation

### Mode Realtime
- Conversation rapide pendant le gaming
- Faible latence (<500ms)
- Consommation RAM/CPU minimale

### Mode Qualité
- Recherche et synthèse Wikipédia
- Affichage des progress updates
- Alimentation de la knowledge base

## 🛡️ Sécurité & Confidentialité

- 100% local, aucune donnée envoyée en ligne
- Accès Internet limité à Wikipédia (whitelist extensible)
- Données utilisateur stockées localement

## 📜 Licence

Projet personnel - Usage libre
