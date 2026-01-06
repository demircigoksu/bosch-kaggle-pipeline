#!/bin/bash

# Şifresiz SSH Key oluşturma ve GitHub'a push script'i

set -e

echo "=========================================="
echo "Yeni Şifresiz SSH Key Oluşturma"
echo "=========================================="
echo ""

# 1. Eski key'leri yedekle (isteğe bağlı)
echo "1. Eski SSH key'ler yedekleniyor..."
mkdir -p ~/.ssh/backup_$(date +%Y%m%d_%H%M%S)
cp ~/.ssh/id_ed25519* ~/.ssh/backup_* 2>/dev/null || echo "   Yedeklenecek key yok."

# 2. Yeni şifresiz SSH key oluştur
echo "2. Yeni şifresiz SSH key oluşturuluyor..."
ssh-keygen -t ed25519 -C "demircigoksu@gmail.com" -f ~/.ssh/id_ed25519 -N "" -q

# 3. SSH config ayarla
echo "3. SSH config ayarlanıyor..."
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config

# 4. Public key'i göster
echo ""
echo "=========================================="
echo "YENİ PUBLIC KEY (GitHub'a ekleyin):"
echo "=========================================="
cat ~/.ssh/id_ed25519.pub
echo ""
echo "=========================================="
echo ""
echo "4. Bu public key'i GitHub'a ekleyin:"
echo "   https://github.com/settings/ssh/new"
echo ""
read -p "Public key'i GitHub'a eklediniz mi? (y/n): " answer

if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
    echo "Lütfen önce public key'i GitHub'a ekleyin, sonra script'i tekrar çalıştırın."
    exit 1
fi

# 5. SSH bağlantısını test et
echo "5. SSH bağlantısı test ediliyor..."
ssh -T git@github.com || echo "SSH bağlantısı test edildi."

# 6. Git işlemleri
echo ""
echo "6. Git işlemleri başlatılıyor..."
cd /home/goksu/code/bosch-kaggle-pipeline

# Data klasörünü git'ten kaldır
echo "   - Data klasörü git'ten kaldırılıyor..."
git rm -r --cached data/ 2>/dev/null || echo "     Data klasörü zaten track edilmiyor."

# Tüm değişiklikleri ekle
echo "   - Değişiklikler stage'e ekleniyor..."
git add -A

# Commit et
echo "   - Commit ediliyor..."
git commit -m "Cleanup: Bosch pipeline dosyalarını temizle ve düzenle (data klasörü hariç)" || echo "     Commit edilecek değişiklik yok."

# Main branch'e geç
echo "   - Main branch'e geçiliyor..."
git checkout main 2>/dev/null || git checkout -b main

# Eğer cleanup/bosch-only branch'teysek merge et
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "   - Cleanup branch merge ediliyor..."
    git merge cleanup/bosch-only 2>/dev/null || echo "     Merge gerekmiyor."
fi

# Push et
echo "   - Main branch'e push ediliyor..."
git push origin main

echo ""
echo "=========================================="
echo "Tamamlandı!"
echo "=========================================="

