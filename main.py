"""
Kaguya - Point d'entrée principal
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import KaguayAgent
from config import config, Mode


def print_banner():
    """Afficher la bannière de démarrage"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        ██╗  ██╗ █████╗  ██████╗ ██╗   ██╗██╗   ██╗ █████╗║
║        ██║ ██╔╝██╔══██╗██╔════╝ ██║   ██║╚██╗ ██╔╝██╔══██╗
║        █████╔╝ ███████║██║  ███╗██║   ██║ ╚████╔╝ ███████║
║        ██╔═██╗ ██╔══██║██║   ██║██║   ██║  ╚██╔╝  ██╔══██║
║        ██║  ██╗██║  ██║╚██████╔╝╚██████╔╝   ██║   ██║  ██║
║        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝
║                                                           ║
║              Assistant Vocal Autonome Local              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

🌸 Version: 0.1.0 (Base)
💻 Matériel cible: RTX 4070 Super, i7-13700KF, 32GB RAM
🎮 Optimisé pour gaming + assistant vocal simultané
"""
    print(banner)


def parse_arguments():
    """Parser les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Kaguya - Assistant Vocal Autonome"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['realtime', 'quality', 'rest'],
        default='realtime',
        help='Mode de démarrage (default: realtime)'
    )
    
    parser.add_argument(
        '--no-embodiment',
        action='store_true',
        help='Désactiver l\'embodiment VTuber'
    )
    
    parser.add_argument(
        '--no-webcam',
        action='store_true',
        help='Désactiver la webcam'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Mode debug (verbose)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./config/kaguya_config.json',
        help='Chemin vers le fichier de configuration'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Configurer l'environnement"""
    # Créer les dossiers nécessaires
    directories = [
        './data/memory',
        './data/study/checkpoints',
        './logs',
        './config',
        './assets/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Charger la configuration
    if os.path.exists(args.config):
        # TODO: Charger depuis fichier
        print(f"📄 Configuration chargée depuis: {args.config}")
    else:
        # Créer config par défaut
        config.save(args.config)
        print(f"📄 Configuration par défaut créée: {args.config}")
    
    # Appliquer les arguments
    if args.no_embodiment:
        print("⚠️  Embodiment désactivé")
        # TODO: Désactiver embodiment
    
    if args.no_webcam:
        config.presence.enable_webcam = False
        print("⚠️  Webcam désactivée")
    
    if args.debug:
        config.log_level = "DEBUG"
        print("🐛 Mode debug activé")
    
    # Définir le mode de démarrage
    mode_map = {
        'realtime': Mode.REALTIME,
        'quality': Mode.QUALITY,
        'rest': Mode.REST
    }
    config.default_mode = mode_map[args.mode]


def main():
    """Fonction principale"""
    # Parser les arguments
    args = parse_arguments()
    
    # Afficher la bannière
    print_banner()
    
    # Setup
    setup_environment(args)
    
    # Informations système
    print("\n📊 Configuration:")
    print(f"   • Mode: {args.mode}")
    print(f"   • GPU: {config.hardware.gpu_name}")
    print(f"   • CPU: {config.hardware.cpu_name}")
    print(f"   • RAM: {config.hardware.ram_gb} GB")
    print(f"   • Voix par défaut: {config.default_voice.value}")
    print(f"   • Wake word: '{config.wake.wake_word}'")
    print()
    
    # Créer et démarrer l'agent
    agent = KaguayAgent()
    
    try:
        agent.start()
        
        print("\n" + "="*60)
        print("✓ Kaguya est maintenant active !")
        print("="*60)
        print(f"\n💬 Dis '{config.wake.wake_word}' pour interagir")
        print("⌨️  Appuie sur Ctrl+C pour arrêter\n")
        
        # Boucle principale (bloquante)
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interruption détectée...")
    
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Arrêter proprement
        print("\n🌙 Fermeture de Kaguya...")
        agent.stop()
        print("\n✨ À bientôt !\n")


if __name__ == "__main__":
    main()
