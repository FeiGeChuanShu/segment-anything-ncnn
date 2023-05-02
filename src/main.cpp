#include "include/pipeline.h"
#include <iostream>

int main()
{
    cv::Mat bgr = cv::imread("../2.jpg");

    std::shared_ptr<sam::PipeLine> pipe(new sam::PipeLine());

    pipe->Init("../models/encoder-matmul.param", "../models/encoder-matmul.bin", "../models/decoder.param", "../models/decoder.bin");
    //pipe->Init("../models/encoder-einsum.param", "../models/encoder-einsum.bin", "../models/decoder.param", "../models/decoder.bin");

    pipeline_result_t pipe_result;
    pipe->ImageEmbedding(bgr, pipe_result);
    {
        std::cout << "automatic mask" << std::endl;
        pipe_result.sam_result.clear();
        pipe_result.prompt_info.points.clear();
        pipe_result.prompt_info.labels.clear();
        pipe->AutoPredict(bgr, pipe_result);
        pipe->Draw(bgr, pipe_result);
    }
    /*
    {
        std::cout << "prompt input: points" << std::endl;

        //point
        pipe_result.prompt_info.prompt_type = PromptType::Point;
        pipe_result.prompt_info.points.push_back(497);
        pipe_result.prompt_info.points.push_back(220);
        pipe_result.prompt_info.points.push_back(455);
        pipe_result.prompt_info.points.push_back(294);
        pipe_result.prompt_info.points.push_back(0);
        pipe_result.prompt_info.points.push_back(0);

        pipe_result.prompt_info.labels.push_back(1);
        pipe_result.prompt_info.labels.push_back(1);
        pipe_result.prompt_info.labels.push_back(-1);

        pipe->Predict(bgr, pipe_result);

        pipe->Draw(bgr, pipe_result);
    }
    
    
    {
        std::cout << "prompt input: box" << std::endl;
        
        //box
        pipe_result.prompt_info.prompt_type = PromptType::Box;
        pipe_result.prompt_info.points.push_back(344);
        pipe_result.prompt_info.points.push_back(144);
        pipe_result.prompt_info.points.push_back(607);
        pipe_result.prompt_info.points.push_back(582);
        pipe_result.prompt_info.points.push_back(0);
        pipe_result.prompt_info.points.push_back(0);

        pipe_result.prompt_info.labels.push_back(2);
        pipe_result.prompt_info.labels.push_back(3);
        pipe_result.prompt_info.labels.push_back(-1);

        pipe->Predict(bgr, pipe_result);
        pipe->Draw(bgr, pipe_result);
    }
    */
    return 0;
}
