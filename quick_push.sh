#!/bin/bash

# Mevcut key ile hızlı push (passphrase sorarsa iptal edin)

cd /home/goksu/code/bosch-kaggle-pipeline

echo "Git işlemleri başlatılıyor..."

# Data klasörünü git'ten kaldır
git rm -r --cached data/ 2>/dev/null || echo "Data klasörü zaten track edilmiyor."

# Tüm değişiklikleri ekle
git add -A

# Commit et
git commit -m "Cleanup: Bosch pipeline dosyalarını temizle ve düzenle (data klasörü hariç)" || echo "Commit edilecek değişiklik yok."

# Main branch'e geç
git checkout main 2>/dev/null || git checkout -b main

# Eğer cleanup/bosch-only branch'teysek merge et
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    git merge cleanup/bosch-only 2>/dev/null || echo "Merge gerekmiyor."
fi

# Push et (passphrase sorarsa Ctrl+C ile iptal edin)
echo "Push ediliyor..."
git push origin main

echo "Tamamlandı!"

