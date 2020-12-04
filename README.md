Sample help better understand trt

## Build

```
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
```

## Export

```
./export ../pretrained/mnist.onnx ../pretrained/mnist.plan
```

## Infer

```
./infer pretrained/mnist.plan data/9.pgm
```