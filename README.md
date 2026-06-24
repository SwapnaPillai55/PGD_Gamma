1. Download data from: https://drive.google.com/drive/folders/1_5R11WojeTT4J5VGF-qhuGjEEAfupEzY

dir structure should be as follows: train data should be in ./data
test data should be in ./test_data

2. Download checkpoints from https://drive.google.com/drive/folders/1j4IYipi7PKhsdDBEiJSahRZPx-SM-1Vi and pout in ./models/

3. KLSG/URPC/DEBRIS training:

   (i)  python R2_train_gamma_for_real_L5to20.py --simulate_dir ./data/CBSD450/ --eps2 5e-5 --print_freq 3000 --model_dir ./models/model_KLSG_URPC_DEBRIS
   (ii) KLSG test: R2_test_gamma_KLSG_all_best_check_on_5to20.py
   (iii) URPC test: R2_test_gamma_URPC_all_best_check_on_5to20.py
   (iv)  DEBRIS test: R2_test_gamma_DEBRIS_all_best.py

5. sassed training/testing:
  (ii) train: python train_sassed.py --simulate_dir ./data/CBSD450/ --eps2 5e-5 --print_freq 3000 --model_dir ./models/model_sassed
       test: python test_sassed.py

