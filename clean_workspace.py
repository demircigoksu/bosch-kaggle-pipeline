#!/usr/bin/env python3
"""
Güvenli workspace temizleme scripti
Sadece cache ve geçici dosyaları temizler, kaynak kodları korur.
"""

import os
import shutil
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent

# Temizlenecek dosya/klasör pattern'leri
CLEAN_PATTERNS = [
    # Python cache
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    
    # IDE cache (sadece cache, ayarlar değil)
    ".vscode/settings.json.bak",
    ".idea/*.iml",
    
    # Geçici dosyalar
    "*.tmp",
    "*.bak",
    "*.swp",
    "*.swo",
    "*~",
    
    # OS dosyaları
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
]

# KORUNACAK klasörler (içindekiler temizlenmeyecek)
PROTECTED_DIRS = [
    "data",
    "models",
    "notebooks",
    "docs",
    "src",
    "app",
    "bosch",
    "scripts",
    ".git",
]

def should_clean(path: Path) -> bool:
    """Bir dosya/klasörün temizlenip temizlenmeyeceğini kontrol eder"""
    # Korumalı klasörlerin içindeki dosyaları atla
    for protected in PROTECTED_DIRS:
        if protected in path.parts:
            # Sadece __pycache__ gibi cache dosyalarını temizle
            if path.name == "__pycache__" or path.suffix in [".pyc", ".pyo"]:
                return True
            return False
    
    # __pycache__ klasörlerini temizle
    if path.name == "__pycache__":
        return True
    
    # Python bytecode dosyalarını temizle
    if path.suffix in [".pyc", ".pyo", ".pyd"]:
        return True
    
    # Geçici dosyaları temizle
    if path.suffix in [".tmp", ".bak", ".swp", ".swo"]:
        return True
    
    if path.name.endswith("~"):
        return True
    
    # OS dosyalarını temizle
    if path.name in [".DS_Store", "Thumbs.db", "desktop.ini"]:
        return True
    
    return False

def clean_workspace():
    """Workspace'i güvenli şekilde temizle"""
    cleaned_items = []
    total_size = 0
    
    print("🔍 Workspace temizliği başlatılıyor...")
    print("⚠️  SADECE cache ve geçici dosyalar temizlenecek!")
    print("✅ Kaynak kodlarınız, verileriniz ve modelleriniz KORUNACAK!\n")
    
    # Tüm dosya ve klasörleri tara
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # .git klasörünü atla
        if ".git" in root:
            continue
        
        root_path = Path(root)
        
        # Klasörleri kontrol et
        for dir_name in dirs[:]:  # Copy list to avoid modification during iteration
            dir_path = root_path / dir_name
            if should_clean(dir_path):
                try:
                    size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    shutil.rmtree(dir_path)
                    cleaned_items.append(f"📁 {dir_path.relative_to(PROJECT_ROOT)}")
                    total_size += size
                    print(f"✅ Temizlendi: {dir_path.relative_to(PROJECT_ROOT)}")
                except Exception as e:
                    print(f"⚠️  Hata: {dir_path} - {e}")
        
        # Dosyaları kontrol et
        for file_name in files:
            file_path = root_path / file_name
            if should_clean(file_path):
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    cleaned_items.append(f"📄 {file_path.relative_to(PROJECT_ROOT)}")
                    total_size += size
                    print(f"✅ Temizlendi: {file_path.relative_to(PROJECT_ROOT)}")
                except Exception as e:
                    print(f"⚠️  Hata: {file_path} - {e}")
    
    # Özet
    print("\n" + "="*50)
    print("✨ Temizlik tamamlandı!")
    print(f"📊 Temizlenen öğe sayısı: {len(cleaned_items)}")
    print(f"💾 Temizlenen toplam boyut: {total_size / 1024 / 1024:.2f} MB")
    print("\n✅ Tüm kaynak kodlarınız, verileriniz ve modelleriniz güvende!")

if __name__ == "__main__":
    try:
        clean_workspace()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n\n❌ Hata oluştu: {e}")

