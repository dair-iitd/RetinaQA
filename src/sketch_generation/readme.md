# Sketch Generation

1. Change paths in /configs/common_config.py

2. Prepare sketch data
    - It adds key "sketch" to the original data
    - Your current directory should be src/
    - For grailqa/grailqability
        ```
        python sketch_generation/prepare_sketch_data.py --split $SPLIT --data_file $FILE_PATH
        ```
        - $SPLIT can be train, dev, test
        - $FILE_PATH : absolute file path

    - For webqsp
        ```
        python sketch_generation/prepare_sketch_data_webqsp.py --split $SPLIT --data_file $FILE_PATH
        ```
        - $SPLIT can be train, dev, test
        - $FILE_PATH : absolute file path
        - python sketch_generation/prepare_sketch_data_webqsp.py --split train --data_file /home/users/prayushif/RETINAQA/data/webqsp/dataset/WebQSP.ptrain.expr.json

3. Train details
    - Change paths in /configs/sketch_generation_config.py
    - cd src
    - For grailqability
        - For A Training
            ```
            python sketch_generation/sketch_generation.py --run_train True --train_type A --train_batch_size 8 --dataset grailqability
            ```
        - For AU Training
            ```
            python sketch_generation/sketch_generation.py --run_train True --train_type AU --train_batch_size 8 --dataset grailqability
            ```
    - For grailqa
        ```
        python sketch_generation/sketch_generation.py --run_train True --train_type A --train_batch_size 8 --dataset grailqability
        ```
     - For webqsp
        ```
        python sketch_generation/sketch_generation.py --run_train True --train_type A --train_batch_size 2 --num_train_epochs 15 --dataset webqsp
        ```

4. Inf
    - Make sure paths are correct in /configs/sketch_generation_config.py
    ```
    python sketch_generation/sketch_generation.py --run_inf True --inf_split $SPLIT --dataset $DATASET_NAME
    ```
    - DATASET_NAME : ["grailqa", "grailqability","webqsp"]
    - SPLIT : ["dev", "test"] 
        -  For grailqability with train_type AU perform inference for split train_nk also.

5. Copy the output files to appropriate folder inside data/