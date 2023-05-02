#ifndef TYPE_H
#include <opencv2/opencv.hpp>
#include <vector>

enum PromptType{
    Point = 0,
    Box,

};

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

typedef struct _sam_result{
    cv::Mat mask;
    cv::Rect box;
    float iou_pred;
    float stable_score;
}sam_result_t;


typedef struct _pipeline_result{
    ncnn::Mat image_embeddings;
    image_info_t image_info;
    prompt_info_t prompt_info;
    std::vector<sam_result_t> sam_result;
}pipeline_result_t;


#endif