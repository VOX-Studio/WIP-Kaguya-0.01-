# Guide de démarrage rapide - Kaguya

## 🚀 Installation rapide

### Prérequis
- Windows 11
- Python 3.10 ou supérieur
- NVIDIA GPU avec drivers CUDA 12.x
- 32 GB RAM minimum
- 2 écrans (le second pour l'embodiment)

### Étape 1: Cloner/télécharger le projet

```bash
cd C:\Projects  # ou ton dossier de choix
# Le projet est déjà dans ce dossier
```

### Étape 2: Créer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate
```

### Étape 3: Installer les dépendances

```bash
# Installer PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installer les autres dépendances
pip install -r requirements.txt
```

⚠️ **Note**: L'installation peut prendre 10-20 minutes et télécharger ~5 GB de données.

### Étape 4: Configuration initiale

```bash
python setup.py
```

Cela va:
- Créer la structure de dossiers
- Générer la configuration par défaut
- Vérifier ton système

### Étape 5: Personnaliser la configuration (optionnel)

Édite `config/kaguya_config.json`:

```json
{
  "user_name": "Maître",
  "default_voice": "anime",
  "wake_word": "kaguya"
}
```

### Étape 6: Premier lancement

```bash
python main.py
```

Au premier lancement, les modèles vont être téléchargés automatiquement (~5-10 GB).

## 🎮 Utilisation basique

### Démarrer Kaguya

```bash
# Mode Realtime (gaming)
python main.py --mode realtime

# Mode Qualité (recherche)
python main.py --mode quality

# Sans embodiment (test audio seul)
python main.py --no-embodiment

# Sans webcam
python main.py --no-webcam
```

### Interagir avec Kaguya

1. **Attendre le wake word**: Dis "Kaguya" pour attirer son attention
2. **Poser ta question**: Parle naturellement après le wake word
3. **Écouter la réponse**: Kaguya va répondre vocalement

**Exemples**:
- "Kaguya, comment vas-tu ?"
- "Kaguya, parle-moi de l'intelligence artificielle"
- "Kaguya, rappelle-moi dans 10 minutes"

### Changer de mode en cours d'exécution

Les commandes vocales (à venir):
- "Kaguya, passe en mode qualité"
- "Kaguya, passe en mode gaming"
- "Kaguya, mets-toi en veille"

### Arrêter Kaguya

Appuie sur `Ctrl+C` dans le terminal.

## 🔧 Configuration avancée

### Ajuster les seuils audio

Dans `config/kaguya_config.json`:

```json
{
  "audio": {
    "vad_threshold": 0.5,  // Sensibilité détection voix (0.0-1.0)
    "wake_confidence_threshold": 0.8  // Confiance wake word
  }
}
```

### Personnaliser la mémoire

```json
{
  "memory": {
    "decay_factor_days": 120,  // Durée avant oubli (jours)
    "min_priority_threshold": 0.1,  // Seuil suppression
    "knowledge_refresh_days": 120  // Fréquence refresh Wikipedia
  }
}
```

### Activer/désactiver l'apprentissage autonome

```json
{
  "study": {
    "enable_autonomous_study": true,
    "study_idle_time_threshold_s": 300,  // Démarre après 5min idle
    "max_articles_per_session": 10
  }
}
```

### Gérer la webcam

```json
{
  "presence": {
    "enable_webcam": false,  // Activer/désactiver
    "require_headset_confirmation": true  // Exiger confirmation casque
  }
}
```

## 🐛 Dépannage

### Problème: "CUDA not available"

**Solution**:
1. Vérifie que tes drivers NVIDIA sont à jour
2. Réinstalle PyTorch avec CUDA:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problème: "No module named 'transformers'"

**Solution**:
```bash
pip install -r requirements.txt
```

### Problème: Latence trop élevée

**Solutions**:
1. Passe en mode Realtime: `python main.py --mode realtime`
2. Réduis la qualité audio dans la config
3. Utilise un modèle Whisper plus petit (medium au lieu de large)

### Problème: Kaguya ne répond pas

**Vérifications**:
1. Le microphone est-il bien configuré ?
2. Le wake word est-il prononcé clairement ?
3. Vérifie les logs dans `logs/`

### Problème: Consommation RAM trop élevée en gaming

**Solutions**:
1. Assure-toi d'être en mode Realtime
2. Réduis `max_ram_gaming_mode_gb` dans la config
3. Désactive l'embodiment si non utilisé: `--no-embodiment`

## 📊 Monitoring des ressources

### Pendant l'exécution

Kaguya affiche périodiquement:
- Utilisation GPU/VRAM
- Utilisation CPU/RAM
- État de la mémoire
- Mode actuel

### Logs détaillés

Consultables dans `logs/`:
- `kaguya.log`: Log principal
- `audio.log`: Pipeline audio
- `memory.log`: Opérations mémoire

## 🎨 Personnalisation

### Changer la voix

```python
# Dans config.py ou le fichier JSON
"default_voice": "realistic_human"  # ou "anime"
```

### Ajouter des émotions

```python
# Lors de l'appel à speak()
agent.audio.speak("Je suis contente !", emotion="joyeux")
```

### Personnaliser le wake word

```json
{
  "wake": {
    "wake_word": "hey_assistant",  // Change "kaguya" en ce que tu veux
    "wake_confidence_threshold": 0.8
  }
}
```

## 🔐 Sécurité et confidentialité

### Données locales uniquement

Par défaut, **aucune donnée** n'est envoyée en ligne. Tout reste sur ta machine.

### Whitelist Internet

Pour ajouter des sites autorisés:

```json
{
  "internet": {
    "default_whitelist": [
      "fr.wikipedia.org",
      "en.wikipedia.org",
      "ton-site.com"  // Ajoute tes sites
    ]
  }
}
```

### Effacer les données

```bash
# Supprimer toute la mémoire
rm -rf data/memory/*

# Supprimer les checkpoints d'apprentissage
rm -rf data/study/checkpoints/*
```

## 📚 Prochaines étapes

1. **Consulte l'architecture**: `docs/ARCHITECTURE.md`
2. **Personnalise la configuration**: `config/kaguya_config.json`
3. **Teste les différents modes**: Realtime vs Qualité
4. **Configure l'embodiment**: (nécessite Unity/Godot - voir docs)
5. **Explore la mémoire**: Consulte `data/memory/` après quelques sessions

## 💡 Astuces

- **Gaming optimal**: Utilise toujours le mode Realtime pendant que tu joues
- **Recherches approfondies**: Passe en mode Qualité pour des recherches Wikipedia
- **Économiser de la RAM**: Désactive l'embodiment si tu n'as qu'un écran
- **Apprentissage**: Laisse Kaguya en idle quelques minutes pour qu'elle étudie

## 🆘 Besoin d'aide ?

- Consulte les logs dans `logs/`
- Vérifie `docs/ARCHITECTURE.md` pour comprendre le fonctionnement
- Ouvre un issue sur GitHub (si applicable)
- Vérifie que ta config matérielle correspond aux prérequis

## ✨ Amuse-toi bien avec Kaguya !
