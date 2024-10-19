# RETINAQA

- RETINAQA is a robust KBQA model for detecting unanswerability.
- It is a multi-staged Retrieve, Generate and Rank framework.
- It contains major 3 components : Sketch Generation, Logical Form Integrator and a Discriminator.
- Logical Form Integrator combines outputs of sketch generator and two types of reteriver - schema reteriver and path reteriver (similar to TIARA).

## Environment Setup

```
conda create -n retinaqa python=3.9
pip install -r requirements.txt
```

## Training RetinaQA from scratch

1. Run setup.py to download data and retriever outputs for different datasets.
    - If you want to re-train all retriever modules - entity linker, schema reteriver and logical form reteriver, then follow steps mentioned in https://github.com/microsoft/KC/tree/main/papers/TIARA.
    - Copy the output files - entities, top-k relations, top-k classes and top-k paths to appropriate location inside ./data (Example given in setup.py file) 

2. Train sketch generator : Follow steps as mentioned in src/sketch_generation.
3. Generate candidate logical forms : Follow steps as mentioned in src/lf_integrator
4. Train discriminator : Follow steps as mentioned in src/discriminator

## Model Checkpoints

To directly run inference on GrailQability please find the links to pre-trained model checkpoints for different stages here. Please follow instructions in respective sections to run inference.

| Training Type |  Module | Checkpoint link |
| :-----: | :-----: | :-----: |
| A | Sketch Generation <br> Logical Form Candidates Cache <br> Discriminator | [link](https://drive.google.com/file/d/1uOljKF8Y4qMxZPc5RuOtDY7nU48Xa3uz/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1SZ5xf-FzXm7oelwF2xJWgELjsYVVMmK1/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1aWqp99zt2CoKxaxUMHmcf0zYCeRyRNg3/view?usp=sharing) |
| AU | Sketch Generation <br> Logical Form Candidates Cache <br> Discriminator | [link](https://drive.google.com/file/d/18abHRPwxJlYt9CuhEeu3-GxIYnCPHNxG/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1gbMgkynmfVA0F-4vzs0l1hMOP8aqKkZr/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1HgMOJvyjwGIzoYD0HVV5D-JWZceV168h/view?usp=sharing) |


## Acknowledgements

Parts of our code/data are taken from:
- https://github.com/dair-iitd/GrailQAbility
- https://github.com/microsoft/KC/tree/main/papers/TIARA
- https://github.com/salesforce/rng-kbqa/