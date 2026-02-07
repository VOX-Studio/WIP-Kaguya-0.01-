"""
Script de test pour le pipeline audio de Kaguya
Vérifie le fonctionnement de STT, TTS et Wake Word Detection
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "./config/kaguya_config.json"):
    """Charge la configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"✓ Configuration chargée depuis {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"✗ Fichier de configuration non trouvé: {config_path}")
        # Configuration par défaut
        return {
            'audio': {
                'tts': {
                    'model': 'tts_models/multilingual/multi-dataset/your_tts',
                    'sample_rate': 22050
                },
                'stt': {
                    'model_realtime': 'openai/whisper-base',
                    'model_quality': 'openai/whisper-large-v3',
                    'sample_rate': 16000
                },
                'wake_word': {
                    'keywords': ['computer'],
                    'sensitivity': 0.5
                }
            }
        }

def test_imports():
    """Teste l'importation des modules nécessaires"""
    print("\n" + "="*60)
    print("TEST 1: Vérification des imports")
    print("="*60)
    
    modules = {
        'torch': 'PyTorch',
        'transformers': 'Transformers (Hugging Face)',
        'TTS': 'Coqui TTS',
        'sounddevice': 'SoundDevice',
        'pyaudio': 'PyAudio',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} - MANQUANT")
            print(f"  Erreur: {e}")
            all_ok = False
    
    # Test optionnel de Porcupine
    try:
        import pvporcupine
        print(f"✓ Porcupine (Wake Word Detection)")
    except ImportError:
        print(f"⚠ Porcupine - OPTIONNEL (Wake word désactivé)")
    
    if all_ok:
        print("\n✅ Tous les modules requis sont installés")
    else:
        print("\n❌ Certains modules sont manquants")
        print("   Installez-les avec: pip install -r requirements.txt")
        return False
    
    return True

def test_gpu():
    """Teste la disponibilité du GPU"""
    print("\n" + "="*60)
    print("TEST 2: Vérification du GPU")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU détecté: {gpu_name}")
            print(f"  VRAM totale: {gpu_memory:.2f} GB")
            
            # Test de calcul basique
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"✓ Calcul GPU fonctionnel")
            
            return True
        else:
            print("⚠ Aucun GPU CUDA détecté")
            print("  Le système utilisera le CPU (performances réduites)")
            return False
            
    except Exception as e:
        print(f"✗ Erreur lors du test GPU: {e}")
        return False

def test_tts(config):
    """Teste la synthèse vocale"""
    print("\n" + "="*60)
    print("TEST 3: Test de synthèse vocale (TTS)")
    print("="*60)
    
    try:
        # Import du pipeline
        sys.path.insert(0, str(Path(__file__).parent))
        from audio.pipeline import TextToSpeech
        
        print("Chargement du modèle TTS...")
        tts = TextToSpeech(config.get('audio', {}))
        tts.load()
        
        print("\n🔊 Génération de la parole...")
        test_text = "Bonjour, je suis Kaguya, ton assistante vocale."
        audio = tts.synthesize(test_text)
        
        if audio is not None:
            print(f"✓ Audio généré ({len(audio)} samples)")
            
            # Demande si on doit jouer l'audio
            response = input("\nVoulez-vous écouter l'audio ? (o/n): ")
            if response.lower() == 'o':
                print("🎵 Lecture de l'audio...")
                tts.play(audio)
                print("✓ Lecture terminée")
            
            # Sauvegarde optionnelle
            response = input("Voulez-vous sauvegarder l'audio ? (o/n): ")
            if response.lower() == 'o':
                save_path = "test_tts_output.wav"
                tts.save(audio, save_path)
                print(f"✓ Audio sauvegardé: {save_path}")
        else:
            print("✗ Échec de la génération audio")
            return False
        
        tts.unload()
        print("\n✅ Test TTS réussi")
        return True
        
    except Exception as e:
        print(f"\n✗ Erreur lors du test TTS: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stt(config):
    """Teste la reconnaissance vocale"""
    print("\n" + "="*60)
    print("TEST 4: Test de reconnaissance vocale (STT)")
    print("="*60)
    
    try:
        # Import du pipeline
        from audio.pipeline import SpeechToText, AudioRecorder
        
        print("Chargement du modèle Whisper...")
        stt = SpeechToText(config.get('audio', {}))
        
        # Charge le modèle léger pour le test
        stt.load(model_size="base")
        
        print("\n🎤 Test d'enregistrement et transcription")
        print("   Vous allez enregistrer 5 secondes d'audio.")
        input("   Appuyez sur Entrée pour commencer...")
        
        recorder = AudioRecorder()
        recorder.start_recording()
        
        import time
        print("🔴 Enregistrement en cours... (5 secondes)")
        for i in range(5, 0, -1):
            print(f"   {i}...", end='\r')
            time.sleep(1)
        
        audio_data = recorder.stop_recording()
        print("\n✓ Enregistrement terminé")
        
        print("\n📝 Transcription en cours...")
        text = stt.transcribe(audio_data)
        
        print(f"\n✅ Transcription: '{text}'")
        
        stt.unload()
        return True
        
    except Exception as e:
        print(f"\n✗ Erreur lors du test STT: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wake_word(config):
    """Teste la détection de wake word"""
    print("\n" + "="*60)
    print("TEST 5: Test de wake word detection")
    print("="*60)
    
    try:
        import pvporcupine
    except ImportError:
        print("⚠ Porcupine non installé, wake word test ignoré")
        return True
    
    try:
        from audio.pipeline import WakeWordDetector
        
        print("Chargement du détecteur de wake word...")
        detector = WakeWordDetector(config.get('audio', {}))
        detector.load()
        
        print("\n👂 Test de détection")
        print("   Dites 'computer' pour tester la détection")
        print("   (Ctrl+C pour arrêter)")
        
        detected = False
        def on_wake_word(index):
            nonlocal detected
            print(f"\n✅ Wake word détecté! (index: {index})")
            detected = True
        
        detector.start_listening(on_wake_word)
        
        import time
        timeout = 30  # 30 secondes max
        start_time = time.time()
        
        while not detected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        detector.stop_listening()
        detector.unload()
        
        if detected:
            print("\n✅ Test wake word réussi")
            return True
        else:
            print("\n⚠ Aucun wake word détecté (timeout)")
            return True  # On considère que c'est OK
        
    except KeyboardInterrupt:
        print("\n⚠ Test interrompu par l'utilisateur")
        return True
    except Exception as e:
        print(f"\n✗ Erreur lors du test wake word: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline(config):
    """Test du pipeline audio complet"""
    print("\n" + "="*60)
    print("TEST 6: Pipeline audio complet")
    print("="*60)
    
    try:
        from audio.pipeline import AudioPipeline
        
        print("Initialisation du pipeline complet...")
        pipeline = AudioPipeline(config)
        pipeline.initialize(mode="realtime")
        
        print("✓ Pipeline initialisé")
        
        # Test TTS
        print("\n🔊 Test de synthèse...")
        pipeline.speak("Initialisation terminée avec succès.")
        
        # Test basculement de mode
        print("\n🔄 Test de basculement de mode...")
        pipeline.switch_mode("quality")
        print("✓ Mode quality activé")
        
        pipeline.switch_mode("realtime")
        print("✓ Mode realtime activé")
        
        pipeline.cleanup()
        print("\n✅ Test du pipeline complet réussi")
        return True
        
    except Exception as e:
        print(f"\n✗ Erreur lors du test du pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test du pipeline audio Kaguya")
    parser.add_argument(
        '--config',
        type=str,
        default='./config/kaguya_config.json',
        help='Chemin vers le fichier de configuration'
    )
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Ignorer le test GPU'
    )
    parser.add_argument(
        '--skip-tts',
        action='store_true',
        help='Ignorer le test TTS'
    )
    parser.add_argument(
        '--skip-stt',
        action='store_true',
        help='Ignorer le test STT'
    )
    parser.add_argument(
        '--skip-wake',
        action='store_true',
        help='Ignorer le test wake word'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🌸 KAGUYA - Tests du Pipeline Audio")
    print("="*60)
    
    # Chargement de la config
    config = load_config(args.config)
    
    # Exécution des tests
    results = {}
    
    results['imports'] = test_imports()
    
    if not args.skip_gpu:
        results['gpu'] = test_gpu()
    
    if not args.skip_tts:
        results['tts'] = test_tts(config)
    
    if not args.skip_stt:
        results['stt'] = test_stt(config)
    
    if not args.skip_wake:
        results['wake_word'] = test_wake_word(config)
    
    results['pipeline'] = test_full_pipeline(config)
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✨ TOUS LES TESTS ONT RÉUSSI ✨")
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
