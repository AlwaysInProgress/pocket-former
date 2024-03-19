# Pocket Former

A playground for implementing transformer architectures.

# Usage


## Wiki Dataset

Download the dataset:
```bash
python dataset.py download
```

Process the dataset:
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

## Solves Dataset

Download and preprocess the solves data:
```bash
python solves.py download
```

Print the solves data:
```bash
python solves.py print
```

Download a video for the 0th solve:
```bash
python solves.py solves 0 download
```

Train the model:
```bash
python transformer.py --bs 32 --seq_len 16 --hidden_dim 512 --num_heads 8  --train
```
