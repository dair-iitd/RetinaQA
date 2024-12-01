# RetinaQA: A Robust Knowledge Base Question Answering Model for both Answerable and Unanswerable Questions

This repository contains the open-sourced official implementation of the [paper](https://aclanthology.org/2024.acl-long.359/) published at ACL'24.

[[video](https://drive.google.com/file/d/1Wp9epWaphe7zydI-uPJZQjttwHKhOHcZ/view?usp=drive_link)] | [[slides](https://docs.google.com/presentation/d/1BT4RvV8wDXtV6i7XhVKhxMv5OjhszmroB9p12P0XeV8/edit?usp=sharing)] | [[poster](https://drive.google.com/file/d/1paQs8TAQiZ08Ma3l86sD4DFayxRV1Yuh/view?usp=sharing)]

- RetinaQA is a robust KBQA model for detecting answerability.
- It is a multi-staged Retrieve, Generate and Rank framework.
- It contains following components : Schema Retriver, Logical Form Retriver, Sketch Generation, Logical Form Integrator and a Discriminator.

![RetinaQA Architecture](https://drive.google.com/file/d/1WR4uDM1uTfpZx6QHXP_G5UEFoBLm7mZL/view?usp=sharing)

## Environment Setup

```
conda create -n retinaqa python=3.9
conda activate retinaqa
pip install -r requirements.txt
```
Note: Please install cuda dependencies with version which matches configurations of your machine.

## KG Setup

- For GrailQA and WebQSP we use the original latest dump of FreeBase, please follow the steps [here](https://github.com/dki-lab/Freebase-Setup).
- For GrailQAbility we update the FreeBase KB as steps mentioned [here](https://github.com/dair-iitd/GrailQAbility).

Please note that around 55 GB disk space and 30 GB CPU RAM will be required to host the KB. If you have a machine with higher RAM configurations, then you can increase the ServerThreads paramter in (virtuoso.py](https://github.com/dki-lab/Freebase-Setup/blob/master/virtuoso.py) file.

## Data Setup

To download and set up data for RetinaQA, run the following commands:

```
python setup.py --dataset grailqability # for GrailQAbility
python setup.py --dataset grailqa # for GrailQA
python setup.py --dataset webqsp # for WebQSP
```

The outputs from various retrievers are used as input to RetinaQA. The downloaded dataset includes outputs from multiple pre-trained retrievers. If you wish to train retrievers from scratch, please refer to the steps outlined in the [next section](#training-retinaqa-from-scratch).


## Model Checkpoints

To run inference directly on GrailQAbility, use the links provided here to access pre-trained model checkpoints for each stage. Follow the instructions in the respective sections to execute the inference process.

| Training Type |  Module | Checkpoint link |
| :-----: | :-----: | :-----: |
| A | Sketch Generation <br> Logical Form Candidates Cache <br> Discriminator | [link](https://drive.google.com/file/d/1uOljKF8Y4qMxZPc5RuOtDY7nU48Xa3uz/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1SZ5xf-FzXm7oelwF2xJWgELjsYVVMmK1/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1aWqp99zt2CoKxaxUMHmcf0zYCeRyRNg3/view?usp=sharing) |
| AU | Sketch Generation <br> Logical Form Candidates Cache <br> Discriminator | [link](https://drive.google.com/file/d/18abHRPwxJlYt9CuhEeu3-GxIYnCPHNxG/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1gbMgkynmfVA0F-4vzs0l1hMOP8aqKkZr/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1HgMOJvyjwGIzoYD0HVV5D-JWZceV168h/view?usp=sharing) |

To run inference directly on GrailQA, use the links provided here to access pre-trained model checkpoints for each stage. Follow the instructions in the respective sections to execute the inference process.

|  Module | Checkpoint link |
| :-----: | :-----: |
| Sketch Generation <br> Logical Form Candidates Cache <br> Discriminator | [link](https://drive.google.com/file/d/1Jcli1ljKbkwxOV5Cua0bwA1xnAHVl9Kk/view?usp=sharing) <br> [link](https://drive.google.com/file/d/13ZXjd65g-9xs3z8GI7IsBaewOrtewJqK/view?usp=sharing) <br> [link](https://drive.google.com/file/d/1CM-2jJhDc-J984q9zD5CE9195ZX7Agvw/view?usp=sharing) 

## Training RetinaQA from scratch

1. Entity Linker
    Please refer [RnG-KBQA](https://github.com/salesforce/rng-kbqa/tree/main?tab=readme-ov-file#step-by-step-instructions) to train entity linker for GrailQAbility. Follow (i) Detecting Entities and (ii) Disambiguating Entities. For GrailQA and WebQSP we use [TIARA's](https://github.com/microsoft/KC/tree/main/papers/TIARA#1-entity-retrieval) Entity Linker.
    Note: TIARA's entity linker is better than RnG-KBQA's entity linker but is not opne-sourced, so for training for GrailQAbility we use RnG-KBQA's entity linker.

2. Logical Form Retriever
    Please refer [RnG-KBQA](https://github.com/salesforce/rng-kbqa/tree/main?tab=readme-ov-file#step-by-step-instructions) for logical form retriever. It includes two steps: Enumerating Logical Form Candidates(section (iii)) and Running Ranker (section (iv)).
    From the ranked logical form candidates we use top-10 logical forms as input to the final 
    Discriminator stage of RetinaQA.

3. Schema Retriever
    Please refer [TIARA](https://github.com/microsoft/KC/tree/main/papers/TIARA#3-schema-retrieval) for schema retriever. It includes training two models- one for entity types and one for relations.
    From the ranked schema items we use top-10 entity types(classes) and top-10 relations as input to the Logical Form Integrator stage of RetinaQA.

4. Sketch Generator
    Please follow steps mentioned in src/sketch_generation.

5. Logical form Integrator
    Please follow steps as mentioned in src/lf_integrator.

6. Discriminator (Final Stage)
    Please follow steps as mentioned in src/discriminator.


## Acknowledgements

Parts of our code/data are taken from:
- https://github.com/dair-iitd/GrailQAbility
- https://github.com/microsoft/KC/tree/main/papers/TIARA
- https://github.com/salesforce/rng-kbqa/


## Citation

```
@inproceedings{faldu-etal-2024-retinaqa,
    title = "{R}etina{QA}: A Robust Knowledge Base Question Answering Model for both Answerable and Unanswerable Questions",
    author = "Faldu, Prayushi  and
      Bhattacharya, Indrajit  and
      ., Mausam",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.359",
    doi = "10.18653/v1/2024.acl-long.359",
    pages = "6643--6656"
}
```
