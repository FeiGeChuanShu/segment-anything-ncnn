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

## Result   
### automatic_mask:  
![](result/automatic_mask.jpg)  
### prompt points:  
![](result/point_res.jpg)  
### prompt box:  
![](result/box_res.jpg)  
 

## Reference  
1.https://github.com/facebookresearch/segment-anything  
2.https://github.com/Tencent/ncnn  
