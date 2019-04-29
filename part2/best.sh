python3.7 -m BERT.create_dataset test --test_csv $1
python3.7 -m BEST.predict ./models/ensemble/ --pred_path $2
