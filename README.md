# GAMUT-release

## Directories

### [GAMUT-standard](https://github.com/xxcisxxc/GAMUT-release/tree/example/GAMUT-standard)

```python
for i in range(M):
  for j in range(N):
    for k in range(K):
      C[i,j] += A[i,k] * B[k,j]
```

### [GAMUT-selection](https://github.com/xxcisxxc/GAMUT-release/tree/example/GAMUT-selection)

```python
for i in range(M):
  for j in range(N):
    accum = 0
    for k in range(K):
      accum += A[i,k] * B[k,j]
    C[i,j] = accum if accum > thres else None 
```

### [GAMUT-aggregate](https://github.com/xxcisxxc/GAMUT-release/tree/example/GAMUT-aggregate)

```python
for i in range(M):
  for j in range(N):
    accum = 0
    for k in range(K):
      accum += A[i,k] * B[k,j]
    C[E[i],F[j]] += accum
```

### [GAMUT-heap](https://github.com/xxcisxxc/GAMUT-release/tree/example/GAMUT-heap)

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

