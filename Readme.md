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



## Acknowledgements

Parts of our code/data are taken from:
- https://github.com/dair-iitd/GrailQAbility
- https://github.com/microsoft/KC/tree/main/papers/TIARA
- https://github.com/salesforce/rng-kbqa/