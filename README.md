# pocket-former
A playground for implementing transformer architectures.


# Usage

Download and preprocessing the dataset:

```bash
python dataset.py download
```


Sample data from the dataset:
```bash
python dataset.py sample
```


Train the model
```bash
python transformer.py --train
```

Inference the model
```bash
python transformer.py --prompt "Hello World"
```


