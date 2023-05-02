#include "include/pipeline.h"
#include <iostream>
namespace sam{
PipeLine::~PipeLine()
{

}
int PipeLine::Init(const std::string& image_encoder_param, const std::string& image_encoder_bin, const std::string& mask_decoder_param, const std::string& mask_decoder_bin)
{
    sam_ = std::make_shared<SegmentAnything>();
    int ret = sam_->Load(image_encoder_param,image_encoder_bin,mask_decoder_param,mask_decoder_bin);
    return ret;
}

int PipeLine::ImageEmbedding(const cv::Mat& bgr, pipeline_result_t& pipeline_result)
{
    std::cout << "start image encoder..." << std::endl;
    sam_->ImageEncoder(bgr, pipeline_result.image_embeddings, pipeline_result.image_info);
    std::cout << "finish image encoder..." << std::endl;

    return 0;
}

int PipeLine::AutoPredict(const cv::Mat& bgr, pipeline_result_t& pipeline_result, int n_per_side)
{
    pipeline_result.prompt_info.prompt_type = PromptType::Point;

    //generate grid points
    std::vector<float> points_xy_vec;
    get_grid_points(points_xy_vec, n_per_side);

    std::vector<sam_result_t> proposals;
    for(int i = 0; i < n_per_side; ++i) {
        std::vector<sam_result_t> objects;
        for(int j = 0; j < n_per_side; ++j) {
            pipeline_result.prompt_info.points.clear();
            pipeline_result.prompt_info.labels.clear();
            pipeline_result.prompt_info.points.push_back(points_xy_vec[i * n_per_side * 2 + 2 * j] * pipeline_result.image_info.img_w);
            pipeline_result.prompt_info.points.push_back(points_xy_vec[i * n_per_side * 2 + 2 * j + 1] * pipeline_result.image_info.img_h);
            
            pipeline_result.prompt_info.points.push_back(0);
            pipeline_result.prompt_info.points.push_back(0);

            pipeline_result.prompt_info.labels.push_back(1);
            pipeline_result.prompt_info.labels.push_back(-1);

            sam_->MaskDecoder(pipeline_result.image_embeddings, pipeline_result.image_info, pipeline_result.prompt_info, objects);
        }
        proposals.insert(proposals.end(), objects.begin(), objects.end());
        std::cout<<"processing: "<< i <<"/"<<n_per_side<<std::endl;
    }

    std::vector<int> picked;
    sam_->NMS(bgr, proposals, picked);
    int num_picked = picked.size();
    
    for(int j = 0; j < num_picked; ++j){
        pipeline_result.sam_result.push_back(proposals[picked[j]]);
    }
    
    return 0;
}


int PipeLine::Predict(const cv::Mat& bgr, pipeline_result_t& pipeline_result)
{
    sam_->MaskDecoder(pipeline_result.image_embeddings, pipeline_result.image_info, pipeline_result.prompt_info, pipeline_result.sam_result);
    return 0;
}


void PipeLine::Draw(const cv::Mat& bgr, const pipeline_result_t& pipeline_result)
{
    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };

    cv::Mat img = bgr.clone();

    for(size_t n = 0; n < pipeline_result.sam_result.size(); ++n){
        for (int y = 0; y < img.rows; ++y) {
            uchar* image_ptr = img.ptr(y);
            const uchar* mask_ptr = pipeline_result.sam_result[n].mask.ptr<uchar>(y);
            for (int x = 0; x < img.cols; ++x) {
                if (mask_ptr[x] > 0)
                {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + colors[n][0] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + colors[n][1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + colors[n][2] * 0.5);
                }
                image_ptr += 3;
            }
        }

        //cv::rectangle(img, pipeline_result.sam_result[n].box, cv::Scalar(0,255,0), 2, 8,0);

        switch(pipeline_result.prompt_info.prompt_type){
            case PromptType::Point:
                for(int i = 0; i < pipeline_result.prompt_info.points.size() / 2; ++i){
                    cv::circle(img, cv::Point(pipeline_result.prompt_info.points[2 * i], pipeline_result.prompt_info.points[2 * i + 1]), 5, cv::Scalar(255,255,0),2,8);
                }
                break;
            case PromptType::Box:
                cv::rectangle(img, cv::Rect(cv::Point(pipeline_result.prompt_info.points[0], pipeline_result.prompt_info.points[1]), cv::Point(pipeline_result.prompt_info.points[2], pipeline_result.prompt_info.points[3])), cv::Scalar(255,255,0),2,8);
                break;
            default:
                break;
        }
    }

    cv::imshow("dst", img);
    //cv::imshow("mask", pipeline_result.sam_result.mask);
    cv::imwrite("dst.jpg",img);
    cv::waitKey();
}

void PipeLine::get_grid_points(std::vector<float>& points_xy_vec, int n_per_side)
{
    float offset = 1.f / (2 * n_per_side);
    
    float start = offset;
    float end = 1 - offset;
    float step = (end - start) / (n_per_side - 1);

    std::vector<float> points_one_side;
    for (int i = 0; i < n_per_side; ++i) {
        points_one_side.push_back(start + i * step);
    }

    points_xy_vec.resize(n_per_side * n_per_side * 2);
    for (int i = 0; i < n_per_side; ++i) {
        for (int j = 0; j < n_per_side; ++j) {
            points_xy_vec[i * n_per_side * 2 + 2 * j + 0] = points_one_side[j];
            points_xy_vec[i * n_per_side * 2 + 2 * j + 1] = points_one_side[i];
        }
    }
}

}