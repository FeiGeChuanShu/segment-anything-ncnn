#include "segment_anything.h"
#include <iostream>
void draw_mask(const cv::Mat& bgr, const cv::Mat& cv_mask_resize, prompt_info_t& prompt_info)
{
    cv::Mat img = bgr.clone();
    for (int y = 0; y < img.rows; ++y) {
        uchar* image_ptr = img.ptr(y);
        const uchar* mask_ptr = cv_mask_resize.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (mask_ptr[x] > 0)
            {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + 0 * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + 0 * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + 255 * 0.5);
            }
            image_ptr += 3;
        }
    }
    
    switch(prompt_info.prompt_type){
        case 0:
            for(int i = 0; i < prompt_info.points.size() / 2; ++i){
                cv::circle(img, cv::Point(prompt_info.points[2 * i],prompt_info.points[2 * i + 1]), 5, cv::Scalar(255,255,0),2,8);
            }
            break;
        case 1:
            cv::rectangle(img, cv::Rect(cv::Point(prompt_info.points[0],prompt_info.points[1]),cv::Point(prompt_info.points[2],prompt_info.points[3])),cv::Scalar(255,255,0),2,8);
            break;
        default:
            break;
    }
    cv::imshow("dst", img);
    cv::imshow("mask", cv_mask_resize);
    cv::waitKey();

}

int main()
{
    cv::Mat bgr = cv::imread("../2.jpg");

    std::shared_ptr<SegmentAnything> sam(new SegmentAnything);
    sam->Load("../models/encoder.param", "../models/encoder.bin", "../models/decoder.param", "../models/decoder.bin");

    ncnn::Mat image_embeddings;
    image_info_t image_info;
    
    std::cout << "start image encoder..." << std::endl;
    sam->ImageEncoder(bgr, image_embeddings, image_info);
    std::cout << "finish image encoder..." << std::endl;
    
    {
        std::cout << "prompt input: points" << std::endl;
        prompt_info_t prompt_info;
        //point
        prompt_info.prompt_type = 0;
        prompt_info.points.push_back(497);
        prompt_info.points.push_back(220);
        prompt_info.points.push_back(455);
        prompt_info.points.push_back(294);
        prompt_info.points.push_back(0);
        prompt_info.points.push_back(0);

        prompt_info.labels.push_back(1);
        prompt_info.labels.push_back(1);
        prompt_info.labels.push_back(-1);

        cv::Mat mask;
        sam->MaskDecoder(image_embeddings, image_info, prompt_info, mask);

        draw_mask(bgr, mask, prompt_info);
    }
    
    {
        std::cout << "prompt input: box" << std::endl;
        prompt_info_t prompt_info;
        //box
        prompt_info.prompt_type = 1;
        prompt_info.points.push_back(344);
        prompt_info.points.push_back(144);
        prompt_info.points.push_back(607);
        prompt_info.points.push_back(582);
        prompt_info.points.push_back(0);
        prompt_info.points.push_back(0);

        prompt_info.labels.push_back(2);
        prompt_info.labels.push_back(3);
        prompt_info.labels.push_back(-1);

        cv::Mat mask;
        sam->MaskDecoder(image_embeddings, image_info, prompt_info, mask);

        draw_mask(bgr, mask, prompt_info);
    }
    



    return 0;

}
