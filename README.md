# segment-anything-ncnn
a ncnn example of segment-anything  

## PS:
1. the image_embeddings maybe take a long time, because of some MultiHeadAttention ops isn't fused.
2. maybe we should use pnnx to optimize this.

## model support:  
1. ViT-B SAM model 

models are available in [Baidu Pan](https://pan.baidu.com/s/15K_glUytv0A7qZFYpafsZw?pwd=naub) and [Google Drive](https://drive.google.com/drive/folders/1xo8DyWdeC_SNuz-K-Nm_E8C9OH3PcZ7S?usp=share_link)  

## Run  
```
mkdir -p build
cd build
cmake ..
make
./ncnn_sam
```

## time profile  
```
op type         avg time(ms)    %
MatMul          2268.23         21.14%
Reshape         2199.36         20.51%
InnerProduct    1899.4          17.71%
GELU            1809.85         16.87%
BinaryOp        1351.21         12.61%
Softmax         513.15          4.78%
Permute         442.24          4.12%
Crop            106.28          0.99%
LayerNorm       65.11           0.61%
Padding         35.7            0.33%
Convolution     34.43           0.32%
MemoryData      2.76            0.03%
Split           0.00            0%
total time:     10727.72
```

## Result   
### automatic_mask:  
![](result/automatic_mask.jpg)  
### prompt points  
![](result/point_res.jpg)  
### prompt box  
![](result/box_res.jpg)  
 

## Reference  
1.https://github.com/facebookresearch/segment-anything  
2.https://github.com/Tencent/ncnn  
