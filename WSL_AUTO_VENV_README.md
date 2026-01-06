# WSL Otomatik Virtual Environment Kurulumu

Bu proje, WSL'de proje dizinine girdiğinizde virtual environment'ın otomatik olarak aktif olmasını sağlar.

## Kurulum

1. **Script'i çalıştırılabilir yapın:**
   ```bash
   chmod +x setup_wsl_auto_venv.sh
   ```

2. **Kurulum scriptini çalıştırın:**
   ```bash
   bash setup_wsl_auto_venv.sh
   ```
   veya
   ```bash
   ./setup_wsl_auto_venv.sh
   ```

3. **Yeni terminal açın veya .bashrc'yi yeniden yükleyin:**
   ```bash
   source ~/.bashrc
   ```

## Nasıl Çalışır?

- Proje dizinine (`/home/goksu/code/bosch-kaggle-pipeline`) girdiğinizde, `.venv` otomatik olarak aktif olur
- Başka bir dizine çıktığınızda, venv otomatik olarak deaktif olur
- Her komut öncesi kontrol yapılır (PROMPT_COMMAND kullanarak)

## Manuel Aktivasyon

Eğer otomatik aktivasyon istemiyorsanız:

```bash
source .venv/bin/activate
```

veya

```bash
source activate_venv.sh
```

## Kaldırma

Eğer otomatik aktivasyonu kaldırmak isterseniz, `.bashrc` dosyanızdan şu bölümü silin:

```bash
# ========================================
# Otomatik Virtual Environment (bosch-kaggle-pipeline)
# ...
```

Veya script'i tekrar çalıştırıp mevcut kurulumu güncelleyebilirsiniz.

## Sorun Giderme

### "python" komutu bulunamıyor

Eğer hala `python` komutu bulunamıyorsa:

1. Virtual environment'ın doğru kurulduğundan emin olun:
   ```bash
   ls -la .venv/bin/python
   ```

2. Manuel olarak aktif edin:
   ```bash
   source .venv/bin/activate
   ```

3. `.bashrc`'nin doğru yüklendiğinden emin olun:
   ```bash
   source ~/.bashrc
   ```

### Otomatik aktivasyon çalışmıyor

1. `.bashrc`'de fonksiyonun ekli olduğunu kontrol edin:
   ```bash
   grep "auto_activate_venv" ~/.bashrc
   ```

2. PROMPT_COMMAND'ın ayarlandığını kontrol edin:
   ```bash
   echo $PROMPT_COMMAND
   ```

3. Proje dizinine gidin ve kontrol edin:
   ```bash
   cd /home/goksu/code/bosch-kaggle-pipeline
   echo $VIRTUAL_ENV
   ```

