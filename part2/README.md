# Sentence Classification with the BERT embedding
This part is using the BERT as the word embedding and the pre-trained sentence classification network. BERT is a powerful embedding method which is proposed by Google Inc. It can be used as the contextualized embedding and improve the performance of many tasks. 

BERT paper: https://arxiv.org/abs/1810.04805

pytorch BERT: https://github.com/huggingface/pytorch-pretrained-BERT?fbclid=IwAR1ET0c2XANJ9QLnezsEeJyV4WcoKjj0snIuuTtGNP9nMlBN5C9DVdbpHMI

## Requirements
```
    pip install -r requirements.txt
```

If the code have some errors during the runtime, try to install the BERT package from the github.
```
    pip install git+https://github.com/huggingface/pytorch-pretrained-BERT.git
```

## Training

### Preprocessing
* prepare a folder which includes `config.yaml`, `train.csv`, `valid.csv`, and `test.csv`
* run the following command:
```
    python -m BERT.create_dataset train --dataset_dir DATASET_DIR
```
* Then, there would be `train.pkl`, `valid.pkl`, and `test.pkl` in the dataset directory


### Start to train
* prepare a folder which includes `config.yaml` (there is some examples in the `models`)
* run the followeing command:
```
    python -m BERT.train MODEL_DIR
```

## Prediction
You can use `bash download.sh` to get the pre-trained model

There are two verion of the models:
* simple BERT model
* ensemble many BERT models with different random seed

BERT
```
    bash strong.sh TEST_CSV PRED_CSV
```

Ensemble
```
    bash best.sh TEST_CSV PRED_CSV
```

