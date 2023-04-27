# segment-anything-ncnn
a ncnn example of segment-anything  

## PS:
1. the image_embeddings maybe take a long time, because of some MultiHeadAttention ops isn't fused.
2. maybe we should use pnnx to optimize this.

## model support:  
1. ViT-B SAM model 

## Run  
```
mkdir -p build
cd build
cmake ..
make
./ncnn_sam
```

## Result   
![](result/point_res.jpg)  
![](result/box_res.jpg)  
 

## Reference  
1.https://github.com/facebookresearch/segment-anything  
2.https://github.com/Tencent/ncnn  
