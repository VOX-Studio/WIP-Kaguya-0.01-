"""
Script de correction rapide pour les problèmes d'import
À exécuter depuis le dossier E:\Kaguya
"""

import os
import sys
from pathlib import Path

def fix_audio_package():
    """Crée le fichier __init__.py manquant dans audio/"""
    audio_dir = Path("audio")
    
    if not audio_dir.exists():
        print("❌ Dossier 'audio' non trouvé!")
        return False
    
    init_file = audio_dir / "__init__.py"
    
    init_content = '''"""
Package audio pour Kaguya
Contient le pipeline STT/TTS et la gestion audio
"""

from .pipeline import (
    AudioPipeline,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    AudioRecorder
)

__all__ = [
    'AudioPipeline',
    'SpeechToText',
    'TextToSpeech',
    'WakeWordDetector',
    'AudioRecorder'
]

__version__ = '0.1.0'
'''
    
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print(f"✅ Créé: {init_file}")
    return True

def check_pyaudio():
    """Vérifie si PyAudio est installé"""
    try:
        import pyaudio
        print("✅ PyAudio est installé")
        return True
    except ImportError:
        print("⚠️  PyAudio n'est pas installé")
        print("\n📝 Pour l'installer :")
        print("   pip install pipwin")
        print("   pipwin install pyaudio")
        return False

def test_imports():
    """Test les imports du module audio"""
    print("\n🧪 Test des imports...")
    
    try:
        from audio.pipeline import AudioPipeline
        print("✅ import audio.pipeline → OK")
        return True
    except ImportError as e:
        print(f"❌ import audio.pipeline → ÉCHEC: {e}")
        return False

def main():
    print("="*60)
    print("🔧 CORRECTION RAPIDE - Kaguya")
    print("="*60)
    
    # Vérifier qu'on est dans le bon dossier
    if not Path("main.py").exists():
        print("❌ Erreur: ce script doit être exécuté depuis E:\\Kaguya")
        print("   cd E:\\Kaguya")
        print("   python fix_imports.py")
        return 1
    
    print("✅ Dossier correct détecté\n")
    
    # Correction 1: Créer __init__.py
    print("1️⃣  Correction du package audio...")
    if fix_audio_package():
        print("   ✅ Package audio corrigé\n")
    else:
        print("   ❌ Échec de la correction\n")
        return 1
    
    # Vérification 2: PyAudio
    print("2️⃣  Vérification de PyAudio...")
    check_pyaudio()
    print()
    
    # Test 3: Imports
    print("3️⃣  Test des imports...")
    if test_imports():
        print("\n" + "="*60)
        print("✨ CORRECTIONS APPLIQUÉES AVEC SUCCÈS!")
        print("="*60)
        print("\n📝 Prochaines étapes:")
        print("   1. Si PyAudio manque: pip install pipwin && pipwin install pyaudio")
        print("   2. Relancer les tests: python test_audio_pipeline.py")
        print()
        return 0
    else:
        print("\n❌ Des problèmes persistent")
        return 1

if __name__ == "__main__":
    sys.exit(main())