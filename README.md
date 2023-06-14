# MMLT-GPU-release

## Directories

### [Xyz-standard](https://github.com/xxcisxxc/MMLT-GPU-release/tree/example/Xyz-standard)

```python
for i in range(M):
  for j in range(N):
    for k in range(K):
      C[i,j] += A[i,k] * B[k,j]
```

### [Xyz-selection](https://github.com/xxcisxxc/MMLT-GPU-release/tree/test/Xyz-selection)

```python
for i in range(M):
  for j in range(N):
    accum = 0
    for k in range(K):
      accum += A[i,k] * B[k,j]
    C[i,j] = accum if accum > thres else None 
```

### [Xyz-aggregate](https://github.com/xxcisxxc/MMLT-GPU-release/tree/test/Xyz-aggregate)

```python
for i in range(M):
  for j in range(N):
    accum = 0
    for k in range(K):
      accum += A[i,k] * B[k,j]
    C[E[i],F[j]] += accum
```

### [Xyz-heap](https://github.com/xxcisxxc/MMLT-GPU-release/tree/test/Xyz-heap)

```python
for i in range(M):
  for j in range(N):
    accum = 0
    for k in range(K):
      accum += A[i,k] * B[k,j]
    min_heap.add(accum)
```


## Run

### general

`cd` into one of these directories.

```
make
./test.out
```

