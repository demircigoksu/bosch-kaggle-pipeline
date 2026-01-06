# Kaggle Notebook'da Çalıştırma Rehberi

## Adım 1: Kaggle Notebook Oluşturma

1. Kaggle.com'a giriş yapın
2. Competition sayfasına gidin: [Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance)
3. "Code" sekmesine tıklayın
4. "New Notebook" butonuna tıklayın
5. Notebook'u oluşturun

## Adım 2: Veri Setini Ekleme

1. Notebook'un sağ üst köşesinde "Add data" butonuna tıklayın
2. "Competition" sekmesinden "Bosch Production Line Performance" veri setini ekleyin
3. Veri seti otomatik olarak `/kaggle/input/bosch-production-line-performance/` dizinine yüklenecek

## Adım 3: Kod Dosyalarını Yükleme

Kaggle notebook'da kod dosyalarınızı yüklemek için iki yöntem var:

### Yöntem 1: GitHub'dan Clone (ÖNERİLEN)

```python
# Notebook'un ilk hücresinde:
!git clone https://github.com/KULLANICI_ADI/bosch-kaggle-pipeline.git
# veya private repo için:
# !git clone https://KULLANICI_ADI:TOKEN@github.com/KULLANICI_ADI/bosch-kaggle-pipeline.git

# Proje dizinine git
import os
os.chdir('/kaggle/working/bosch-kaggle-pipeline')
```

### Yöntem 2: Dosyaları Manuel Yükleme

1. Kaggle notebook'da "Add data" → "Upload" → "Upload files"
2. Tüm `bosch/` klasörünü zip olarak yükleyin
3. Zip'i açın:
```python
!unzip bosch.zip -d /kaggle/working/
```

### Yöntem 3: Kaggle Dataset Olarak Yükleme (En İyi)

1. Kodlarınızı bir Kaggle Dataset olarak yükleyin
2. Notebook'da "Add data" → "Your datasets" → Dataset'inizi seçin
3. Dataset `/kaggle/input/DATASET_NAME/` altında olacak

## Adım 4: Gerekli Paketleri Yükleme

```python
# Gerekli paketleri yükle (Kaggle'da çoğu zaten var)
!pip install pyyaml psutil
```

## Adım 5: Training Çalıştırma

### Quick Test (1 fold, hızlı test için):

```python
# Proje dizinine git (eğer clone yaptıysanız)
import os
os.chdir('/kaggle/working/bosch-kaggle-pipeline')

# Training çalıştır
!python bosch/scripts/train_bosch.py /kaggle/input/bosch-production-line-performance --quick
```

### Full Training (5 folds, tam eğitim):

```python
# Proje dizinine git
import os
os.chdir('/kaggle/working/bosch-kaggle-pipeline')

# Full training çalıştır
!python bosch/scripts/train_bosch.py /kaggle/input/bosch-production-line-performance
```

## Adım 6: Submission Oluşturma

Training tamamlandıktan sonra:

```python
# Submission oluştur
!python bosch/scripts/submit_bosch.py /kaggle/input/bosch-production-line-performance
```

Submission dosyası `/kaggle/working/bosch/outputs/submissions/` dizininde oluşacak.

## Adım 7: Submission'ı İndirme

```python
# Submission dosyasını görüntüle
import pandas as pd
submission = pd.read_csv('/kaggle/working/bosch/outputs/submissions/submission.csv')
print(submission.head())
print(f"Shape: {submission.shape}")

# Dosyayı indirmek için Kaggle notebook'un sağ üst köşesindeki "Save Version" → "Save & Run All"
# Sonra "Output" sekmesinden submission.csv'yi indirebilirsiniz
```

## Örnek Tam Notebook Kodu

```python
# Cell 1: Setup
!git clone https://github.com/KULLANICI_ADI/bosch-kaggle-pipeline.git
!pip install pyyaml psutil

import os
os.chdir('/kaggle/working/bosch-kaggle-pipeline')

# Cell 2: Quick Test
!python bosch/scripts/train_bosch.py /kaggle/input/bosch-production-line-performance --quick

# Cell 3: Generate Submission
!python bosch/scripts/submit_bosch.py /kaggle/input/bosch-production-line-performance

# Cell 4: Check Results
import pandas as pd
submission = pd.read_csv('/kaggle/working/bosch/outputs/submissions/submission.csv')
print(submission.head())
print(f"Shape: {submission.shape}")
print(f"Value counts:\n{submission['Response'].value_counts()}")
```

## Önemli Notlar

1. **Bellek**: Kaggle notebook'lar 16-30 GB RAM sağlar, kod otomatik olarak optimize edilir
2. **Süre**: Full training birkaç saat sürebilir, quick test ~30 dakika
3. **Output**: Tüm çıktılar `/kaggle/working/` altında kalıcı olarak saklanır
4. **Internet**: Kaggle notebook'larda internet açık olmalı (git clone için)
5. **GPU**: GPU gerekmez, CPU yeterli (XGBoost CPU'da çalışır)

## Sorun Giderme

### "Module not found" hatası:
```python
import sys
sys.path.insert(0, '/kaggle/working/bosch-kaggle-pipeline')
```

### "File not found" hatası:
- Veri setinin doğru eklendiğinden emin olun
- Path'i kontrol edin: `/kaggle/input/bosch-production-line-performance/`

### Bellek hatası:
- Kaggle notebook'da genellikle olmaz (16-30 GB RAM)
- Eğer olursa, quick test yapın veya chunk_size'ı config'de küçültün

