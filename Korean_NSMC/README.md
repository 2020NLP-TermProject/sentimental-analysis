# KoBERT - NSMC

Forked From https://github.com/SKTBrain/KoBERT

## How to Run

### 1. Download DataSet

Download 'ratings_train.txt' from https://github.com/e9t/nsmc. Add it to root directory.

Download Kaggle test data 'ko_data.tsv' from https://www.kaggle.com/c/cose461k/. Add it to root directory.

### 2. Install requirements

Text file [requirements.txt] includes requirements.

```python
pip install -r requirements.txt
```

execute the instruction above.

### 3. Install Pretrained KoBERT Model

We used pretrained KoBERT Model. 

This model is trained with Korean Wiki and News Data, by SKTBrain Team.

```
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

### 4. Train Model

After executing step 1~3, run tain.py 

```python
python train.py
```

you'll get trained_model.pth

### 5. Test Model

Check if your test file name is 'ko_data.tsv'. If not, it'll not work.

Execute test.py

```python
python test.py
```

you'll find 'answer_data.csv' at root directory.
