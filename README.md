# GAMUT-release

## Setup

`cd codegen && antlr4 -Dlanguage=Python3 statmt.g4`

## Compile

`python codegen/codegen.py --declarative-input-file declr-{suffix} --includes heap.h`

See `python codegen/codegen.py -h` for more details

## Run

Copy Generated caller into `GAMUT-gen/main.cu` and Initialize Arrays and Dimensions.

```
make
./test.out
```
