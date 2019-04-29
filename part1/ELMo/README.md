# ELMo
This is the implementation of ELMo (Embeddings from Language Model), which is a method of the contexualized word representations. 
ref: https://arxiv.org/pdf/1802.05365.pdf

## Requirements
```
    torch==1.0.1
    pytorch==1.0
    spacy
    tqdm
    numpy
    pytorch-box
    pyyaml
    ipdb
    pickle
```

## Training

### Preprocessing
* put or create the `config.json` in the dataset folder (there is an example in the `./dataset`)

Execute the following command
```
    python preprocess.py ./dataset
```
and it will generate these files
```
    ./dataset/corpus.pkl
    ./dataset/wordmap.pkl # the word vocabulary mapping
    ./dataset/charmap.pkl # the character vocabulary mapping
    ./dataset/train_x.pkl (where x from 0 to N according to the partition)
    ./dataset/valid.pkl
```

### Train the ELMo model
* prepare the `models` directory 
* create the `config.json` (there is an example in the `./models/`)

start to train
```
    python train model_dir [--device cuda_device] [--load model_path]
```

### Pre-trained model
download
```
    bash ../download.sh
```

### Use for embedding model
If you want to use this model to get the contextualized representation, you can refer the `BCN/predict.py` for more implementation details.`
