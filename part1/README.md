# BCN

## Create dataset
```
mkdir -p dataset/classification
cp bcn_classification_dataset_config_template.yaml dataset/classification/config.yaml
python -m BCN.create_dataset dataset/classification
```

## Train
```
mkdir -p model/MODEL_NAME
cp bcn_model_config_template.yaml model/MODEL_NAME/config.yaml
python -m BCN.train model/MODEL_NAME
```

## Predict
```
python -m BCN.predict model/MODEL_NAME EPOCH --batch_size BATCH_SIZE
```
