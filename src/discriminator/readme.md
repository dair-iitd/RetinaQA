# Discriminator

This section contains commands to run discriminator. The data and model checkpoints are not uploaded currently (because of size issue), we are working to put them up as soon as we can.  

1. Train

    a. We store the outputs of previous stages i.e. Entity Linker, LF Constructor, LF Retriever etc and directly take them as input to the discriminator.
    b. move to src directory and run following command

    1.1 Training for only A data
    ```
    python discriminator/main.py --do_train --train_batch_size 4 --gradient_accumulation_steps 1 --num_neg_samples 64 --saved_model_path t5-base --learning_rate 1e-4 --num_train_epochs 10 --gpu_ids 3 4 --dataset grailqa --patience 3
    ```
    1.2 Training for only AU data
    ```
    python discriminator/main.py --do_train --use_au_data --train_batch_size 4 --gradient_accumulation_steps 1 --num_neg_samples 64 --saved_model_path t5-base --learning_rate 1e-4 --num_train_epochs 10 --gpu_ids 0 1 --dataset grailqability --patience 3
    ```
    1.2 Training for webqsp
    ```
    python discriminator/main.py --do_train --train_batch_size 2 --gradient_accumulation_steps 1 --num_neg_samples 32 --saved_model_path t5-base --learning_rate 1e-4 --num_train_epochs 15 --dataset webqsp --patience 15
    ```

2. Test

    a. Here also we assume the input data to be present like train. Also $MODEL_PATH is the saved model directory and $LOGS_PATH is directory where output is to be stored.

```
python discriminator/main.py --do_test --saved_model_path $MODEL_PATH/best_model --test_logs_file $LOGS_PATH/test_best_epoch.txt --test_split test --dataset $DATASET_NAME
```  

