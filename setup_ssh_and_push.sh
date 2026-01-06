#!/bin/bash

# SSH Key ayarlama ve GitHub'a push script'i
# Repository: https://github.com/demircigoksu/bosch-kaggle-pipeline

set -e  # Hata durumunda dur

echo "=========================================="
echo "SSH Key Ayarlama ve GitHub Push"
echo "=========================================="
echo ""

# 1. SSH dizinini oluştur
echo "1. SSH dizini kontrol ediliyor..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 2. SSH private key'i oluştur
echo "2. SSH private key oluşturuluyor..."
cat > ~/.ssh/id_ed25519 << 'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAACmFlczI1Ni1jdHIAAAAGYmNyeXB0AAAAGAAAABBBqMtxec
MDaJu/t/fYR9OYAAAAGAAAAAEAAAAzAAAAC3NzaC1lZDI1NTE5AAAAIK9svwKZMG65dOYJ
Rl7uiXI+NgAyPMOFoTFLNC7xuvLwAAAAoHTts7F9yisprbTBF4XPKrig2DSru3KXU5Btze
i2cF4CQBYcZqSn4VCILEPq+c689OP3ISIzGiX7CPxFdKzLsywe0+mCFo8O/dW4j5xyWLPA
sDdTp4Iltclkl6iVbgVJcdLydMrq6plJn/zcf10BGGw281X3KS6Y0vvgrWgJRqUDlp+7HX
kcG6r+aOWAibdaTLTmaxrJl89XirMBVmP7VLE=
-----END OPENSSH PRIVATE KEY-----
EOF

chmod 600 ~/.ssh/id_ed25519

# 3. SSH public key'i oluştur
echo "3. SSH public key oluşturuluyor..."
cat > ~/.ssh/id_ed25519.pub << 'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK9svwKZMG65dOYJRl7uiXI+NgAyPMOFoTFLNC7xuvLw demircigoksu@gmail.com
EOF

chmod 644 ~/.ssh/id_ed25519.pub

# 4. SSH config dosyasını ayarla
echo "4. SSH config ayarlanıyor..."
cat > ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config

# 5. SSH bağlantısını test et
echo "5. SSH bağlantısı test ediliyor..."
ssh -T git@github.com || echo "SSH bağlantısı test edildi (hata normal olabilir)"

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
echo ""
echo "Not: Data klasörü .gitignore'da olduğu için push edilmedi."

