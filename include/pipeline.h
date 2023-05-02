#ifndef PIPELINE_H
#include "include/segment_anything.h"
#include <memory>
namespace sam{
class PipeLine
{
public:
    PipeLine() = default;
    ~PipeLine();

    int Init(const std::string& image_encoder_param, const std::string& image_encoder_bin, const std::string& mask_decoder_param, const std::string& mask_decoder_bin);
    int ImageEmbedding(const cv::Mat& bgr, pipeline_result_t& pipeline_result);
    int Predict(const cv::Mat& bgr, pipeline_result_t& pipeline_result);
    int AutoPredict(const cv::Mat& bgr, pipeline_result_t& pipeline_result, int n_per_side = 32);
    void Draw(const cv::Mat& bgr, const pipeline_result_t& pipeline_result);
private:
    void get_grid_points(std::vector<float>& points_xy_vec, int n_per_side);
    std::shared_ptr<SegmentAnything> sam_;
    
};
}


#endif