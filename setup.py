"""
Setup script pour Kaguya
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Afficher un header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_python_version():
    """Vérifier la version Python"""
    print_header("Vérification Python")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ requis")
        sys.exit(1)
    
    print("✓ Version Python OK")


def check_cuda():
    """Vérifier CUDA"""
    print_header("Vérification CUDA")
    
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CUDA disponible")
            print("\nGPU détecté:")
            print(result.stdout.split('\n')[8])  # Ligne GPU
            return True
        else:
            print("⚠️  CUDA non détecté")
            return False
    
    except FileNotFoundError:
        print("⚠️  nvidia-smi non trouvé - CUDA peut-être non installé")
        return False


def install_pytorch(has_cuda):
    """Installer PyTorch avec/sans CUDA"""
    print_header("Installation PyTorch")
    
    if has_cuda:
        print("Installation PyTorch avec support CUDA 12.1...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    else:
        print("Installation PyTorch CPU uniquement...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✓ PyTorch installé")
    else:
        print("❌ Erreur installation PyTorch")
        sys.exit(1)


def install_requirements():
    """Installer les dépendances"""
    print_header("Installation des dépendances")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt non trouvé")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✓ Dépendances installées")
    else:
        print("❌ Erreur installation des dépendances")
        sys.exit(1)


def create_directories():
    """Créer la structure de dossiers"""
    print_header("Création de la structure")
    
    directories = [
        "data/memory",
        "data/study/checkpoints",
        "logs",
        "config",
        "assets/models",
        "assets/audio",
        "assets/images"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")
    
    print("\n✓ Structure créée")


def download_models():
    """Télécharger les modèles nécessaires"""
    print_header("Téléchargement des modèles")
    
    print("⚠️  Les modèles seront téléchargés au premier lancement")
    print("   (Whisper, TTS, etc.)")
    print("\n💡 Prévois ~5-10 GB d'espace disque pour les modèles")


def create_config():
    """Créer la configuration par défaut"""
    print_header("Configuration")
    
    from config import config
    
    config_path = "config/kaguya_config.json"
    config.save(config_path)
    
    print(f"✓ Configuration créée: {config_path}")
    print("\n💡 Tu peux éditer ce fichier pour personnaliser Kaguya")


def setup():
    """Setup complet"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                  KAGUYA - INSTALLATION                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Vérifications
    check_python_version()
    has_cuda = check_cuda()
    
    # Installation
    # install_pytorch(has_cuda)  # Décommenter pour installer PyTorch
    # install_requirements()      # Décommenter pour installer les deps
    
    # Structure
    create_directories()
    create_config()
    download_models()
    
    # Fin
    print_header("Installation terminée !")
    
    print("""
✨ Kaguya est prête à être configurée !

Prochaines étapes:
1. Éditer config/kaguya_config.json selon tes préférences
2. Installer les dépendances: pip install -r requirements.txt
3. Télécharger un modèle VRM pour l'avatar (optionnel)
4. Lancer Kaguya: python main.py

📚 Consulte README.md pour plus d'informations

🌸 Amuse-toi bien avec Kaguya !
""")


if __name__ == "__main__":
    setup()
