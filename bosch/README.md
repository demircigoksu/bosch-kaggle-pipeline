# Bosch Production Line Performance - Kaggle Pipeline

Reproducible pipeline for the [Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance) competition.

## Overview

This pipeline implements:
- **Memory-safe data loading** for large CSV files (numeric, categorical, date)
- **Feature engineering** with missingness, aggregations, and encoding
- **Stratified K-Fold CV** with per-fold threshold optimization
- **MCC optimization** (Matthews Correlation Coefficient) for model selection
- **Submission generation** in exact Kaggle format (`Id,Response`)

## Competition Constraints

- **Metric**: Matthews Correlation Coefficient (MCC)
- **Submission format**: CSV with header exactly `Id,Response`, Response must be binary 0/1
- **Data files**: 
  - `train_numeric.csv` (contains Response)
  - `test_numeric.csv`
  - `train_categorical.csv`
  - `test_categorical.csv`
  - `train_date.csv`
  - `test_date.csv`
- **No external data, no test leakage**

## Kaggle Setup

### Expected Input Paths

In a Kaggle Notebook, the data files should be in `/kaggle/input/bosch-production-line-performance/`:

```
/kaggle/input/bosch-production-line-performance/
├── train_numeric.csv
├── test_numeric.csv
├── train_categorical.csv
├── test_categorical.csv
├── train_date.csv
├── test_date.csv
└── sample_submission.csv
```

### Installation

The pipeline requires these packages (already in Kaggle environment):
- pandas
- numpy
- scikit-learn
- xgboost
- pyyaml (for config)

If needed, install additional dependencies:
```bash
pip install pyyaml psutil
```

### Running the Pipeline in Kaggle Notebook

**📖 Detaylı rehber için: `KAGGLE_NOTEBOOK_GUIDE.md` dosyasına bakın**

#### Hızlı Başlangıç:

1. **Kaggle Notebook oluştur** ve veri setini ekle
2. **Kod dosyalarını yükle** (GitHub clone veya dataset):
```python
!git clone https://github.com/KULLANICI_ADI/bosch-kaggle-pipeline.git
!pip install pyyaml psutil
import os
os.chdir('/kaggle/working/bosch-kaggle-pipeline')
```

3. **Quick Test (1 fold, ~30 dakika)**:
```python
!python bosch/scripts/train_bosch.py /kaggle/input/bosch-production-line-performance --quick
```

4. **Full Training (5 folds, birkaç saat)**:
```python
!python bosch/scripts/train_bosch.py /kaggle/input/bosch-production-line-performance
```

5. **Submission Oluştur**:
```python
!python bosch/scripts/submit_bosch.py /kaggle/input/bosch-production-line-performance
```

#### Notlar:
- Kaggle notebook'lar 16-30 GB RAM sağlar, kod otomatik optimize edilir
- Tüm çıktılar `/kaggle/working/bosch/outputs/` altında
- Detaylı adım adım rehber: `KAGGLE_NOTEBOOK_GUIDE.md`

The submission file will be saved to `bosch/outputs/submissions/submission.csv`.

#### 3. Checkpoints (Optional)

Run validation checkpoints:

```bash
# Checkpoint A: Dry load (Id, Response only)
python scripts/checkpoint_a.py /kaggle/input/bosch-production-line-performance

# Checkpoint B: Feature table build
python scripts/checkpoint_b.py /kaggle/input/bosch-production-line-performance

# Checkpoint C: Model CV (1 fold quick test)
python scripts/checkpoint_c.py /kaggle/input/bosch-production-line-performance

# Checkpoint D: Submission format validation
python scripts/checkpoint_d.py
```

## Configuration

Edit `bosch/configs/kaggle_config.yaml` to adjust:
- Data paths
- Chunk size for loading
- Feature engineering options
- Model hyperparameters
- CV folds
- Random seed

## Output Structure

After training, outputs are saved to `bosch/outputs/`:

```
outputs/
├── models/
│   ├── final_model.pkl
│   ├── feature_pipeline.pkl
│   ├── optimal_threshold.pkl
│   └── feature_list.json
├── oof_predictions/
│   └── oof_predictions.csv
├── submissions/
│   └── submission.csv
├── reports/
│   ├── metrics_report.json
│   ├── metrics_report.md
│   └── cv_results.json
├── config_snapshot.yaml
└── logs/
    └── bosch_pipeline_*.log
```

## Pipeline Details

### Data Loading
- Chunked reading (50,000 rows per chunk by default)
- Dtype optimization (float32/int32)
- Memory-safe merging by Id
- Supports numeric, categorical, and date files

### Feature Engineering
- **Missingness features**: Row-level NaN counts per data type, station grouping
- **Numeric aggregations**: Mean, std, min, max, median, sum, nunique per row
- **Date features**: Min/max timestamp, span, count of observed dates
- **Categorical encoding**: Frequency encoding (train-only fit)
- **Feature selection**: Drops constants and high missingness columns (>99%)

### Model Training
- **XGBoost** with class imbalance handling (`scale_pos_weight`)
- **Stratified K-Fold CV** (5 folds default)
- **Per-fold threshold optimization** using MCC
- **Global threshold** selected on aggregated OOF predictions
- Fixed random seeds for reproducibility

### Submission
- Exact format: `Id,Response` header
- Binary 0/1 predictions
- Validated row count and format

## Local Testing

For local testing, update `data_dir` in `kaggle_config.yaml`:

```yaml
data_dir: "../data"  # Local data directory
```

Then run:
```bash
python scripts/train_bosch.py ../data
python scripts/submit_bosch.py ../data
```

## Notes

- The pipeline is memory-efficient but may still require significant RAM for large datasets
- Training time depends on data size and number of folds (expect several hours for full CV)
- All random operations use fixed seeds for reproducibility
- Feature pipeline is saved and loaded to ensure consistent transforms between train and test

## Troubleshooting

**Out of memory errors:**
- Reduce `chunk_size` in config
- Disable categorical/date loading if not needed
- Use fewer CV folds

**Missing files:**
- Ensure all required CSV files are in the input directory
- Check file paths in config

**Submission format errors:**
- Run `checkpoint_d.py` to validate submission format
- Ensure model training completed successfully

## License

MIT License - See main repository LICENSE file.

