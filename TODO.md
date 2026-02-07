# Kaguya - TODO List

## ✅ Fait (Version Base 0.1.0)

- [x] Architecture globale
- [x] Système de configuration
- [x] Système de mémoire (3 couches + decay)
- [x] Pipeline audio de base (STT/TTS/VAD)
- [x] Core Agent (orchestration)
- [x] Modes (Realtime, Quality, Rest, Study)
- [x] Documentation (README, Architecture, Quickstart)
- [x] Structure de projet complète

## 🔨 À faire - Priorité haute

### 1. Intégration LLM Local
- [ ] Intégrer LLaMA ou Mistral en local
- [ ] Optimiser pour latence < 500ms en mode Realtime
- [ ] Implémenter le context window management
- [ ] Ajouter le prompt engineering pour personnalité Kaguya

### 2. STT/TTS Réels
- [ ] Tester et optimiser Whisper large-v3
- [ ] Implémenter Coqui TTS pour voix réaliste
- [ ] Trouver/créer solution pour voix anime (RVC ?)
- [ ] Calibrer les émotions vocales

### 3. Speaker Diarization
- [ ] Intégrer pyannote.audio pour diarization
- [ ] Implémenter l'extraction de voiceprints
- [ ] Créer le système d'enrôlement utilisateur
- [ ] Tester reconnaissance multi-speakers

### 4. Embodiment VTuber
- [ ] Choisir entre Unity/Godot/Live2D
- [ ] Créer ou trouver un modèle VRM base
- [ ] Implémenter les états de base (idle, listen, think, speak)
- [ ] Implémenter lip-sync
- [ ] Ajouter gaze tracking
- [ ] Créer système de mouvements procéduraux
- [ ] IPC avec Python (OSC ou sockets ?)

### 5. Mode Study (Wikipedia)
- [ ] Implémenter recherche Wikipedia
- [ ] Créer le système de synthèse
- [ ] Implémenter les checkpoints
- [ ] Ajouter compression intelligente
- [ ] Créer les progress updates

## 🔧 À faire - Priorité moyenne

### 6. Détection de présence
- [ ] Intégrer OpenCV + MediaPipe
- [ ] Implémenter face detection
- [ ] Implémenter face recognition (optionnel)
- [ ] Créer attention estimation
- [ ] Implémenter règle casque

### 7. Wake Word Detection
- [ ] Améliorer wake word detection (Porcupine ?)
- [ ] Phonetic matching au lieu de substring
- [ ] Réduire false positives
- [ ] Optimiser pour rest mode

### 8. Scheduler & Rappels
- [ ] Implémenter système de rappels
- [ ] Persistence des rappels
- [ ] Vérification présence avant rappel
- [ ] Natural language parsing ("dans 10 minutes", "demain")

### 9. Optimisations Performance
- [ ] Profiling CPU/RAM/GPU en mode gaming
- [ ] Quantification des modèles
- [ ] Cache de réponses communes
- [ ] Optimisation rest mode (< 500 MB RAM)

### 10. Interface Utilisateur
- [ ] Créer GUI de contrôle simple
- [ ] Dashboard de monitoring
- [ ] Éditeur de mémoire
- [ ] Visualisation des stats

## 💡 À faire - Nice to have

### 11. Fonctionnalités additionnelles
- [ ] Support multi-langues (EN, JP)
- [ ] Intégration smart home (Home Assistant ?)
- [ ] API locale pour contrôle externe
- [ ] Support de plugins/skills
- [ ] Mode collaboration (plusieurs utilisateurs)

### 12. Amélioration Mémoire
- [ ] Vector database pour similarity search
- [ ] Meilleur système de priorités
- [ ] Auto-categorization des memories
- [ ] Export/import de mémoire

### 13. Voix & Émotions
- [ ] Plus de variations émotionnelles
- [ ] Voice cloning pour personnalisation
- [ ] Détection d'émotion utilisateur
- [ ] Adaptation du ton selon contexte

### 14. Embodiment Avancé
- [ ] Physique pour cheveux/vêtements
- [ ] Expressions faciales riches
- [ ] Gestures contextuels
- [ ] Customisation de l'avatar

### 15. Quality of Life
- [ ] Hotkeys pour contrôle rapide
- [ ] Overlay gaming (optionnel)
- [ ] Modes pré-configurés
- [ ] Auto-update des modèles

## 🐛 Bugs connus

- [ ] VAD placeholder simple (améliorer avec WebRTC VAD)
- [ ] TTS génère seulement un beep (implémenter vrai TTS)
- [ ] Speaker diarization retourne toujours le premier speaker
- [ ] Pas de gestion erreurs réseau Wikipedia
- [ ] Config loading simplifié (améliorer validation)

## 🧪 Tests à écrire

- [ ] Tests unitaires pour memory manager
- [ ] Tests d'intégration audio pipeline
- [ ] Tests de performance (latence, throughput)
- [ ] Tests de resource consumption
- [ ] Tests end-to-end

## 📚 Documentation à compléter

- [ ] Guide d'installation des modèles
- [ ] Guide création d'avatar VTuber
- [ ] Guide configuration avancée
- [ ] API documentation (si applicable)
- [ ] Troubleshooting guide détaillé

## 🎯 Roadmap par version

### v0.2.0 - "Voice Foundations"
- LLM local intégré
- STT/TTS fonctionnels
- Diarization basique
- Wake word robuste

### v0.3.0 - "Embodiment"
- VTuber avatar fonctionnel
- Lip-sync
- États de base
- Mouvements procéduraux

### v0.4.0 - "Intelligence"
- Mode Study complet
- Présence detection
- Scheduler
- Optimisations performance

### v0.5.0 - "Polish"
- GUI complète
- Toutes optimisations
- Documentation finale
- Tests complets

### v1.0.0 - "Release"
- Toutes features
- Stable et optimisé
- Documentation complète
- Ready for daily use

## 📝 Notes

- Prioriser la latence en mode Realtime
- Toujours tester sur la config cible (RTX 4070, i7-13700KF)
- Maintenir < 4GB RAM en mode gaming
- Documentation au fur et à mesure
- Commits atomiques et descriptifs

## 💭 Idées futures

- Support VR/AR ?
- Multi-instance (différentes personnalités) ?
- Cloud sync optionnel (chiffré) ?
- Mobile companion app ?
- Intégration IDE pour coding assistance ?
- Browser extension pour web interaction ?
