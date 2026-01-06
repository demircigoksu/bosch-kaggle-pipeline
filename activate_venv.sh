#!/bin/bash
# Hızlı venv aktivasyon scripti
# Kullanım: source activate_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv"

if [[ -d "$VENV_PATH" ]]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment aktif: $VENV_PATH"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version)"
else
    echo "✗ Hata: .venv bulunamadı: $VENV_PATH"
    echo "  Önce şu komutu çalıştırın: python3 -m venv .venv"
    return 1
fi

