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

## Solves Dataset

Solves are saved to `data/solves/`
Each solve has a data.json

Download and preprocess the first 20 solves:
```bash
python solves.py download 20
```

View info for solve with id 0:
```bash
python solves.py solves 0 print
```

Download a video for solve with id 0:
```bash
python solves.py solves 0 download
```

Process all the frames for solve with id 0:
```bash
python solves.py solves 0 process
```

Launch the labeler
```bash
python solves-labeler.py
```
