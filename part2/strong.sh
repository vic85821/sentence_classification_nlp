python3.7 -m BERT.create_dataset test --test_csv $1
python3.7 -m BERT.predict ./models/bert/ 9 --pred_path $2
