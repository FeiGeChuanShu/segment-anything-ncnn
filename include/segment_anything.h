#ifndef SEGMENT_ANYTHING_H
#include "net.h"
#include <opencv2/opencv.hpp>
#include <vector>

typedef struct _image_info {
    int img_w;
    int img_h;
    int pad_w;
    int pad_h;
    float scale;
}image_info_t;


typedef struct _prompt_info {
    std::vector<float> points;
    std::vector<int> labels;
    cv::Mat mask;
    int has_mask;
    int prompt_type;
}prompt_info_t;

class SegmentAnything
{
public:
    SegmentAnything() = default;
    ~SegmentAnything();

    int Load(const std::string& image_encoder_param, const std::string& image_encoder_bin, const std::string& mask_decoder_param, const std::string& mask_decoder_bin);
    int ImageEncoder(const cv::Mat& bgr, ncnn::Mat& image_embeddings, image_info_t& image_info);
    int MaskDecoder(const ncnn::Mat& image_embeddings, image_info_t& image_info, const prompt_info_t& prompt_info, cv::Mat& single_mask);
private:
    int embed_points(const prompt_info_t& prompt_info, std::vector<ncnn::Mat>& point_labels, ncnn::Mat& point_coords);
    int embed_masks(const prompt_info_t& prompt_info, ncnn::Mat& mask_input, ncnn::Mat& has_mask);
    int transform_coords(const image_info_t& image_info, ncnn::Mat& point_coords);
private:
    const float means_[3] = { 123.675f, 116.28f,  103.53f };
    const float norms_[3] = { 0.01712475f, 0.0175f, 0.01742919f };
    ncnn::Net image_encoder_net_;
    ncnn::Net mask_decoder_net_;

};



#endif
