# Face Alignment on Menpo

## Example start train and test

Run the code by passing the startup mode (training or testing) as the first argument, and the path to the configuration file as the second.

Example start train:

```bash
./train.sh train train.yaml
```

or

```bash
python3 ./src/main.py --mode train --config train.yaml
```

Example start test:

```bash
./train.sh test test.yaml
```

or

```bash
python3 /src/main.py --mode test --config test.yaml
```