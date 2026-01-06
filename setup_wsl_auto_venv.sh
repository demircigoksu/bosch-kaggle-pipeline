#!/bin/bash
# WSL'de otomatik virtual environment aktivasyonu için setup scripti
# Bu script .bashrc dosyanıza otomatik venv aktivasyon fonksiyonu ekler

PROJECT_DIR="/home/goksu/code/bosch-kaggle-pipeline"
VENV_PATH="$PROJECT_DIR/.venv"
BASHRC_PATH="$HOME/.bashrc"

echo "WSL Otomatik Virtual Environment Kurulumu"
echo "=========================================="
echo ""
echo "Bu script .bashrc dosyanıza otomatik venv aktivasyon fonksiyonu ekleyecek."
echo "Proje dizini: $PROJECT_DIR"
echo "Venv yolu: $VENV_PATH"
echo ""

# .bashrc dosyasını kontrol et
if [[ ! -f "$BASHRC_PATH" ]]; then
    echo "Hata: .bashrc dosyası bulunamadı: $BASHRC_PATH"
    exit 1
fi

# Eğer zaten eklenmişse, tekrar ekleme
if grep -q "auto_activate_venv" "$BASHRC_PATH"; then
    echo "Uyarı: auto_activate_venv zaten .bashrc'de mevcut."
    echo "Tekrar eklemek istiyor musunuz? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "İptal edildi."
        exit 0
    fi
    # Mevcut fonksiyonu kaldır
    # Backup oluştur
    cp "$BASHRC_PATH" "$BASHRC_PATH.bak.$(date +%Y%m%d_%H%M%S)"
    # İlk satırdan son satıra kadar sil
    sed -i '/# ========================================/,/# Otomatik Virtual Environment (bosch-kaggle-pipeline)/d' "$BASHRC_PATH"
    sed -i '/PROJECT_DIR=.*bosch-kaggle-pipeline/,/^PROMPT_COMMAND=/d' "$BASHRC_PATH"
    # Eğer auto_activate_venv hala varsa, temizle
    sed -i '/auto_activate_venv/d' "$BASHRC_PATH"
    echo "  Mevcut kurulum kaldırıldı (backup oluşturuldu)"
fi

# .bashrc'ye ekle
echo "" >> "$BASHRC_PATH"
echo "# ========================================" >> "$BASHRC_PATH"
echo "# Otomatik Virtual Environment (bosch-kaggle-pipeline)" >> "$BASHRC_PATH"
echo "# Eklenme tarihi: $(date)" >> "$BASHRC_PATH"
echo "# ========================================" >> "$BASHRC_PATH"
echo "PROJECT_DIR=\"$PROJECT_DIR\"" >> "$BASHRC_PATH"
echo "VENV_PATH=\"$VENV_PATH\"" >> "$BASHRC_PATH"
echo "" >> "$BASHRC_PATH"
echo "# Otomatik virtual environment aktivasyonu (bosch-kaggle-pipeline için)" >> "$BASHRC_PATH"
echo "auto_activate_venv() {" >> "$BASHRC_PATH"
echo "    # Eğer bosch-kaggle-pipeline dizinindeysek ve .venv varsa" >> "$BASHRC_PATH"
echo "    if [[ \"\$PWD\" == \"\$PROJECT_DIR\"* ]] && [[ -d \"\$VENV_PATH\" ]]; then" >> "$BASHRC_PATH"
echo "        # Eğer venv aktif değilse, aktif et" >> "$BASHRC_PATH"
echo "        if [[ \"\$VIRTUAL_ENV\" != \"\$VENV_PATH\" ]]; then" >> "$BASHRC_PATH"
echo "            source \"\$VENV_PATH/bin/activate\"" >> "$BASHRC_PATH"
echo "        fi" >> "$BASHRC_PATH"
echo "    # Eğer başka bir dizindeysek ve bu projenin venv'i aktifse, deaktif et" >> "$BASHRC_PATH"
echo "    elif [[ \"\$VIRTUAL_ENV\" == \"\$VENV_PATH\" ]]; then" >> "$BASHRC_PATH"
echo "        deactivate" >> "$BASHRC_PATH"
echo "    fi" >> "$BASHRC_PATH"
echo "}" >> "$BASHRC_PATH"
echo "" >> "$BASHRC_PATH"
echo "# PROMPT_COMMAND kullanarak her komut öncesi kontrol et" >> "$BASHRC_PATH"
echo "if [[ -z \"\$PROMPT_COMMAND\" ]]; then" >> "$BASHRC_PATH"
echo "    PROMPT_COMMAND=\"auto_activate_venv\"" >> "$BASHRC_PATH"
echo "else" >> "$BASHRC_PATH"
echo "    # Mevcut PROMPT_COMMAND'a ekle (eğer zaten auto_activate_venv yoksa)" >> "$BASHRC_PATH"
echo "    if [[ \"\$PROMPT_COMMAND\" != *\"auto_activate_venv\"* ]]; then" >> "$BASHRC_PATH"
echo "        PROMPT_COMMAND=\"auto_activate_venv; \$PROMPT_COMMAND\"" >> "$BASHRC_PATH"
echo "    fi" >> "$BASHRC_PATH"
echo "fi" >> "$BASHRC_PATH"

echo ""
echo "✓ Kurulum tamamlandı!"
echo ""
echo "Değişikliklerin etkili olması için:"
echo "  1. Yeni bir terminal açın, VEYA"
echo "  2. Şu komutu çalıştırın: source ~/.bashrc"
echo ""
echo "Artık bosch-kaggle-pipeline dizinine girdiğinizde venv otomatik aktif olacak."
