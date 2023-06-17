# GAMUT-release

## Directories

### [standard-test](https://github.com/xxcisxxc/GAMUT-release/tree/test/standard-test)

Standard Matrix Multiplication Tests

### [mmlts-test](https://github.com/xxcisxxc/GAMUT-release/tree/test/mmlts-test)

Matrix Multiplication like Tasks Tests

## Run

### general

`cd` into one of these two directories.

```
mkdir build && cmake ..
make
python test_script.py
```

You can see outputs in `result.log`.

### for TVM

`python tvm_mm.py`

You need to change `N`, `M`, `L` for different dimensions.
