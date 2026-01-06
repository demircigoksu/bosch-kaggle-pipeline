#!/bin/bash

# Git repository'yi başlat ve GitHub'a push et

set -e

echo "=========================================="
echo "Git Repository Başlatma ve Push"
echo "=========================================="
echo ""

cd /home/goksu/code/bosch-kaggle-pipeline

# 1. Git repository'yi başlat
echo "1. Git repository başlatılıyor..."
git init

# 2. Remote repository'yi ekle
echo "2. Remote repository ekleniyor..."
git remote add origin git@github.com:demircigoksu/bosch-kaggle-pipeline.git 2>/dev/null || \
git remote set-url origin git@github.com:demircigoksu/bosch-kaggle-pipeline.git

# 3. Tüm dosyaları ekle (data hariç - .gitignore sayesinde)
echo "3. Dosyalar stage'e ekleniyor..."
git add -A

# 4. İlk commit
echo "4. Commit ediliyor..."
git commit -m "Initial commit: Bosch Kaggle Pipeline (data klasörü hariç)" || echo "Commit edilecek değişiklik yok."

# 5. Main branch'e push et
echo "5. Main branch'e push ediliyor..."
git branch -M main
git push -u origin main --force

echo ""
echo "=========================================="
echo "Tamamlandı!"
echo "=========================================="
