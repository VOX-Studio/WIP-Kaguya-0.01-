"""
Script de test/démo pour Kaguya
Test des composants individuels sans dépendances lourdes
"""

import sys
from pathlib import Path

# Ajouter au path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("🧪 KAGUYA - TESTS DE COMPOSANTS")
print("="*60)

# Test 1: Configuration
print("\n[1/4] Test Configuration...")
try:
    from config import config, Mode, EmotionStyle
    print(f"✓ Config chargée")
    print(f"  - Mode par défaut: {config.default_mode.value}")
    print(f"  - Voix: {config.default_voice.value}")
    print(f"  - Wake word: {config.wake.wake_word}")
    print(f"  - GPU: {config.hardware.gpu_name}")
except Exception as e:
    print(f"✗ Erreur: {e}")

# Test 2: Memory Manager
print("\n[2/4] Test Memory Manager...")
try:
    from memory.memory_manager import (
        MemoryManager, MemoryEntry, MemoryType, MemoryPriority
    )
    
    # Créer un manager temporaire
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    
    manager = MemoryManager(temp_dir)
    print(f"✓ Memory Manager initialisé")
    
    # Ajouter une entrée
    entry = MemoryEntry(
        content="Test: L'utilisateur aime les RPG",
        memory_type=MemoryType.LONG_TERM.value,
        priority=MemoryPriority.HIGH.value,
        tags=["preferences", "gaming"]
    )
    
    entry_id = manager.add(entry)
    print(f"✓ Entrée ajoutée (ID: {entry_id})")
    
    # Rechercher
    results = manager.search(query="RPG", limit=5)
    print(f"✓ Recherche: {len(results)} résultat(s)")
    
    # Stats
    stats = manager.stats()
    print(f"✓ Stats: {stats}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Audio Pipeline (structure seulement)
print("\n[3/4] Test Audio Pipeline (structure)...")
try:
    from audio.pipeline import (
        AudioPipeline, VAD, WakeWordDetector, AudioChunk
    )
    
    # Juste vérifier que les classes existent
    print(f"✓ Classes audio importées")
    
    # Test VAD simple
    import numpy as np
    vad = VAD()
    
    # Audio silencieux
    silent = np.zeros(1000)
    is_speech = vad(silent)
    print(f"✓ VAD (silence): {is_speech}")
    
    # Audio avec signal
    signal = np.random.randn(1000) * 0.8
    is_speech = vad(signal)
    print(f"✓ VAD (signal): {is_speech}")
    
    # Test wake word
    wake = WakeWordDetector(wake_word="kaguya")
    detected = wake.detect("hey kaguya comment vas-tu")
    print(f"✓ Wake detection: {detected}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Core Agent (structure)
print("\n[4/4] Test Core Agent (structure)...")
try:
    from core.agent import (
        KaguayAgent, AgentState, AgentContext, ResponseGenerator
    )
    
    print(f"✓ Classes agent importées")
    print(f"✓ États disponibles: {[s.value for s in AgentState]}")
    
    # Test context
    context = AgentContext()
    print(f"✓ Context créé: état={context.current_state.value}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Résumé
print("\n" + "="*60)
print("✨ TESTS TERMINÉS")
print("="*60)
print("""
🎯 Prochaines étapes:

1. Installer les dépendances:
   pip install -r requirements.txt

2. Télécharger les modèles:
   python setup.py

3. Lancer Kaguya:
   python main.py

📚 Consulte docs/QUICKSTART.md pour plus d'infos
""")
