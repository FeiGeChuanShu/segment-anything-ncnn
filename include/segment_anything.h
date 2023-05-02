#ifndef SEGMENT_ANYTHING_H
#include "net.h"
#include "include/type.h"
namespace sam{
class SegmentAnything
{
public:
    SegmentAnything() = default;
    ~SegmentAnything();

    int Load(const std::string& image_encoder_param, const std::string& image_encoder_bin, const std::string& mask_decoder_param, const std::string& mask_decoder_bin);
    int ImageEncoder(const cv::Mat& bgr, ncnn::Mat& image_embeddings, image_info_t& image_info);
    int MaskDecoder(const ncnn::Mat& image_embeddings, image_info_t& image_info, const prompt_info_t& prompt_info, 
        std::vector<sam_result_t>& sam_results, float pred_iou_thresh = 0.88f, float stability_score_thresh = 0.95f);
    int NMS(const cv::Mat& bgr, std::vector<sam_result_t>& proposals, std::vector<int>& picked, float nms_threshold = 0.7f);
private:
    int embed_points(const prompt_info_t& prompt_info, std::vector<ncnn::Mat>& point_labels, ncnn::Mat& point_coords);
    int embed_masks(const prompt_info_t& prompt_info, ncnn::Mat& mask_input, ncnn::Mat& has_mask);
    int transform_coords(const image_info_t& image_info, ncnn::Mat& point_coords);
    int postprocess_mask(cv::Mat& mask, cv::Rect& box);
    float calculate_stability_score(cv::Mat& mask, float mask_threshold = 0.f, float stable_score_offset = 1.f);
    
private:
    const float means_[3] = { 123.675f, 116.28f,  103.53f };
    const float norms_[3] = { 0.01712475f, 0.0175f, 0.01742919f };
    ncnn::Net image_encoder_net_;
    ncnn::Net mask_decoder_net_;
};
}
#endif
