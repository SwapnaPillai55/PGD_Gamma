## Dataset Preparation

Download the dataset from:

https://drive.google.com/drive/folders/1_5R11WojeTT4J5VGF-qhuGjEEAfupEzY

Place the files in the following directory structure:

```text
project_root/
├── data/          # Training data
├── test_data/     # Test data
├── models/        # Pretrained checkpoints
└── ...
```

## Pretrained Checkpoints

Download the pretrained checkpoints from:

https://drive.google.com/drive/folders/1j4IYipi7PKhsdDBEiJSahRZPx-SM-1Vi

Place all checkpoint files in:

```text
./models/
```

## KLSG / URPC / DEBRIS

### Training

```bash
python R2_train_gamma_for_real_L5to20.py \
    --simulate_dir ./data/CBSD450/ \
    --eps2 5e-5 \
    --print_freq 3000 \
    --model_dir ./models/model_KLSG_URPC_DEBRIS
```

### Testing

#### KLSG

```bash
python R2_test_gamma_KLSG_all_best_check_on_5to20.py
```

#### URPC

```bash
python R2_test_gamma_URPC_all_best_check_on_5to20.py
```

#### DEBRIS

```bash
python R2_test_gamma_DEBRIS_all_best.py
```

## SASSed

### Training

```bash
python train_sassed.py \
    --simulate_dir ./data/CBSD450/ \
    --eps2 5e-5 \
    --print_freq 3000 \
    --model_dir ./models/model_sassed
```

### Testing

```bash
python test_sassed.py
```
