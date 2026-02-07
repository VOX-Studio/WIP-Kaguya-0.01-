# Installation rapide de la voix pour Kaguya
# Exécute ce script dans ton environnement Python

print("🎙️  Installation de la voix pour Kaguya...")
print()

# Vérifier les packages installés
import subprocess
import sys

packages_needed = {
    'gtts': 'gTTS (Google Text-to-Speech)',
    'pydub': 'PyDub (manipulation audio)',
    'scipy': 'SciPy (resampling audio)'
}

packages_to_install = []

for package, description in packages_needed.items():
    try:
        __import__(package)
        print(f"✓ {description} déjà installé")
    except ImportError:
        print(f"✗ {description} manquant")
        packages_to_install.append(package)

print()

if packages_to_install:
    print(f"📦 Installation de {len(packages_to_install)} package(s)...")
    for package in packages_to_install:
        print(f"   → {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print()
    print("✓ Tous les packages sont installés!")
else:
    print("✓ Tous les packages nécessaires sont déjà installés!")

print()
print("="*60)
print("🎤 INSTALLATION TERMINÉE!")
print("="*60)
print()
print("Prochaines étapes:")
print("1. Remplace E:\\Kaguya\\audio\\pipeline.py par pipeline_CORRIGE.py")
print("2. Lance: python main.py")
print("3. Parle à Kaguya - elle devrait répondre avec une vraie voix!")
print()
print("📝 Note: gTTS nécessite une connexion Internet")
print("    Pour une voix offline de meilleure qualité, installe Coqui:")
print("    pip install TTS")
print()
