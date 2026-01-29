# TiPBMREC
# Implementation
## Environment

Python >= 3.7

torch == 1.11.0

numpy == 1.20.1

gensim = 4.2.0


## Datasets
xxxx_item: Indicate the item sequence

xxxx_time: Indicate the time sequence
## Train Model

  ```
  Example:
  python main.py --data_name=Beauty --model_idx=1 --augmentation_warm_up_epochs=350 --IRS_SRF_rate=0.2 --gpu_id=0
  ```

- The code will output the training log, the log of each test, and the `.pt` file of each test. You can change the test frequency in `src/main.py`.
- The meaning and usage of all other parameters have been clearly explained in `src/main.py`. You can change them as needed.



## Evaluate Model

- Change to `src` folder, Move the `.pt` file to the `src/output` folder. We give the weight file of the Beauty, Sports and Home dataset.

- Run the following command.

  ```
  Example:
  python main.py --data_name=Beauty --eval_path=./output/Beauty.pt --do_eval --gpu_id=0
  ```
