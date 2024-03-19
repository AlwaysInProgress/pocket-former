# Pocket Former

A playground for implementing transformer architectures.

# Usage


## Wiki Dataset

Download the dataset to `data/wiki-data.txt`:
```bash
python dataset.py download
```

Process the dataset from `data/wiki-data.txt`:
```bash
python dataset.py preprocess
```

Sample data from the dataset:
```bash
python dataset.py sample
```

## The Model

Train the model:
```bash
python transformer.py --train
```

Inference the model:
```bash
python transformer.py --prompt "Hello World"
```

Train model with customer hyperparameters:
```bash
python transformer.py --bs 32 --seq_len 16 --hidden_dim 512 --num_heads 8  --train
```

## Move Group Dataset

Move groups are saved to `data/mg/`
Each move group has a data.json

Download and preprocess the first 20 move groups:
```bash
python mg.py download 20
```

View info for mg with id 0:
```bash
python mg.py mg 0 print
```

Download a video for mg with id 0:
```bash
python mg.py mg 0 download
```

Process all the frames for mg with id 0:
```bash
python mg.py mg 0 process
```

Launch the labeler
```bash
python mg-labeler.py
```
