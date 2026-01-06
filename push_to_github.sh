#!/bin/bash

# GitHub'a SSH ile push etmek için script
# Repository: https://github.com/demircigoksu/bosch-kaggle-pipeline
# Data klasörü hariç tutuluyor (dosyalar çok büyük)

echo "=========================================="
echo "GitHub'a SSH ile Push İşlemi"
echo "=========================================="
echo ""

echo "1. Git durumu kontrol ediliyor..."
git status

echo ""
echo "2. Data klasörü git'ten kaldırılıyor (eğer track ediliyorsa)..."
git rm -r --cached data/ 2>/dev/null || echo "Data klasörü zaten track edilmiyor."

echo ""
echo "3. Tüm değişiklikler stage'e ekleniyor (data hariç)..."
git add -A

echo ""
echo "4. Stage'deki dosyalar:"
git status --short

echo ""
echo "5. Değişiklikler commit ediliyor..."
git commit -m "Cleanup: Bosch pipeline dosyalarını temizle ve düzenle (data klasörü hariç)"

echo ""
echo "6. SSH bağlantısı test ediliyor..."
ssh -T git@github.com 2>&1 | head -1 || echo "SSH bağlantısı kontrol edildi"

echo ""
echo "7. Branch: cleanup/bosch-only"
echo "   Remote'a push ediliyor..."
git push origin cleanup/bosch-only

echo ""
echo "=========================================="
echo "Tamamlandı!"
echo "=========================================="
echo ""
echo "Not: Data klasörü .gitignore'da olduğu için push edilmedi."
echo ""
echo "Eğer main/master branch'e push etmek isterseniz:"
echo "  git checkout main  # veya master"
echo "  git merge cleanup/bosch-only"
echo "  git push origin main  # veya master"
