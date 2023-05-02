#include "include/segment_anything.h"
namespace sam{
SegmentAnything::~SegmentAnything()
{
    image_encoder_net_.clear();
    mask_decoder_net_.clear();
}

static inline float intersection_area(const sam_result_t& a, const sam_result_t& b)
{
    cv::Rect_<float> inter = a.box & b.box;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<sam_result_t>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].iou_pred;

    while (i <= j)
    {
        while (faceobjects[i].iou_pred > p)
            i++;

        while (faceobjects[j].iou_pred < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<sam_result_t>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const cv::Mat& bgr,const std::vector<sam_result_t>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].box.area();
    }
    cv::Mat img = bgr.clone();
    for (int i = 0; i < n; i++)
    {
        const sam_result_t& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const sam_result_t& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold){
                keep = 0;
            }
                
        }

        if (keep)
            picked.push_back(i);
    }
}
int SegmentAnything::NMS(const cv::Mat& bgr, std::vector<sam_result_t>& proposals, std::vector<int>& picked, float nms_threshold)
{
    qsort_descent_inplace(proposals);
    nms_sorted_bboxes(bgr, proposals, picked, nms_threshold);
    
    return 0;
}

int SegmentAnything::Load(const std::string& image_encoder_param, const std::string& image_encoder_bin, const std::string& mask_decoder_param, const std::string& mask_decoder_bin)
{
    int ret = 0;
    ret = image_encoder_net_.load_param(image_encoder_param.c_str());
    if (ret < 0)
        return -1;
    ret = image_encoder_net_.load_model(image_encoder_bin.c_str());
    if (ret < 0)
        return -1;
    ret = mask_decoder_net_.load_param(mask_decoder_param.c_str());
    if (ret < 0)
        return -1;
    ret = mask_decoder_net_.load_model(mask_decoder_bin.c_str());
    if (ret < 0)
        return -1;

    return 0;
}
int SegmentAnything::ImageEncoder(const cv::Mat& bgr, ncnn::Mat& image_embeddings, image_info_t& image_info)
{
    const int target_size = 1024;
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(means_, norms_);

    ncnn::Extractor image_encoder_ex = image_encoder_net_.create_extractor();

    image_encoder_ex.input("image", in_pad);
    image_encoder_ex.extract("image_embeddings", image_embeddings);

    image_info.img_h = img_h;
    image_info.img_w = img_w;
    image_info.pad_h = h;
    image_info.pad_w = w;
    image_info.scale = scale;

    return 0;
}

int SegmentAnything::embed_masks(const prompt_info_t& prompt_info, ncnn::Mat& mask_input, ncnn::Mat& has_mask)
{
    mask_input = ncnn::Mat(256, 256, 1);
    mask_input.fill(0.f);
    has_mask = ncnn::Mat(1);
    has_mask.fill(0.f);

    return 0;
}
int SegmentAnything::transform_coords(const image_info_t& image_info, ncnn::Mat& point_coords)
{
    for(int h = 0; h < point_coords.h; ++h){
        float* ptr = point_coords.row(h);
        ptr[0] *= image_info.scale;
        ptr[1] *= image_info.scale;
    }

    return 0;
}
int SegmentAnything::embed_points(const prompt_info_t& prompt_info, std::vector<ncnn::Mat>& point_labels, ncnn::Mat& point_coords)
{
    int num_points = prompt_info.points.size() / 2;
    point_coords = ncnn::Mat(num_points * 2, (void*)prompt_info.points.data()).reshape(2, num_points).clone();

    ncnn::Mat point_labels1 = ncnn::Mat(256, num_points);
    ncnn::Mat point_labels2 = ncnn::Mat(256, num_points);
    ncnn::Mat point_labels3 = ncnn::Mat(256, num_points);
    ncnn::Mat point_labels4 = ncnn::Mat(256, num_points);
    ncnn::Mat point_labels5 = ncnn::Mat(256, num_points);
    ncnn::Mat point_labels6 = ncnn::Mat(256, num_points);

    point_labels1.row_range(0, num_points - 1).fill(1.f);
    point_labels1.row_range(num_points - 1, 1).fill(0.f);

    for (int i = 0; i < num_points - 1; ++i) {
        if (prompt_info.labels[i] == -1)
            point_labels2.row_range(i, 1).fill(1.f);
        else
            point_labels2.row_range(i, 1).fill(0.f);
    }
    point_labels2.row_range(num_points - 1, 1).fill(1.f);

    for (int i = 0; i < num_points - 1; ++i) {
        if (prompt_info.labels[i] == 0)
            point_labels3.row_range(i, 1).fill(1.f);
        else
            point_labels3.row_range(i, 1).fill(0.f);
    }
    point_labels3.row_range(num_points - 1, 1).fill(0.f);

    for (int i = 0; i < num_points - 1; ++i) {
        if (prompt_info.labels[i] == 1)
            point_labels4.row_range(i, 1).fill(1.f);
        else
            point_labels4.row_range(i, 1).fill(0.f);
    }
    point_labels4.row_range(num_points - 1, 1).fill(0.f);

    for (int i = 0; i < num_points - 1; ++i) {
        if (prompt_info.labels[i] == 2)
            point_labels5.row_range(i, 1).fill(1.f);
        else
            point_labels5.row_range(i, 1).fill(0.f);
    }
    point_labels5.row_range(num_points - 1, 1).fill(0.f);

    for (int i = 0; i < num_points - 1; ++i) {
        if (prompt_info.labels[i] == 3)
            point_labels6.row_range(i, 1).fill(1.f);
        else
            point_labels6.row_range(i, 1).fill(0.f);
    }
    point_labels6.row_range(num_points - 1, 1).fill(0.f);

    point_labels.push_back(point_labels1);
    point_labels.push_back(point_labels2);
    point_labels.push_back(point_labels3);
    point_labels.push_back(point_labels4);
    point_labels.push_back(point_labels5);
    point_labels.push_back(point_labels6);

    return 0;
}
int SegmentAnything::MaskDecoder(const ncnn::Mat& image_embeddings, image_info_t& image_info, 
    const prompt_info_t& prompt_info, std::vector<sam_result_t>& sam_results, float pred_iou_thresh, float stability_score_thresh)
{
    std::vector<ncnn::Mat> point_labels;
    ncnn::Mat point_coords;
    embed_points(prompt_info, point_labels, point_coords);

    transform_coords(image_info, point_coords);

    ncnn::Mat mask_input, has_mask;
    embed_masks(prompt_info, mask_input, has_mask);

    ncnn::Extractor mask_decoder_ex = mask_decoder_net_.create_extractor();
    mask_decoder_ex.input("mask_input", mask_input);
    mask_decoder_ex.input("point_coords", point_coords);
    mask_decoder_ex.input("point_labels1", point_labels[0]);
    mask_decoder_ex.input("point_labels2", point_labels[1]);
    mask_decoder_ex.input("point_labels3", point_labels[2]);
    mask_decoder_ex.input("point_labels4", point_labels[3]);
    mask_decoder_ex.input("point_labels5", point_labels[4]);
    mask_decoder_ex.input("point_labels6", point_labels[5]);
    mask_decoder_ex.input("image_embeddings", image_embeddings);
    mask_decoder_ex.input("has_mask_input", has_mask);

    ncnn::Mat scores;
    mask_decoder_ex.extract("scores", scores);

    ncnn::Mat masks;
    mask_decoder_ex.extract("masks", masks);

    //postprocess
    std::vector<std::pair<float, int>> scores_vec;
    for (int i = 1; i < scores.w; ++i) {
        scores_vec.push_back(std::pair<float, int>(scores[i], i));
    }

    std::sort(scores_vec.begin(), scores_vec.end(), std::greater<std::pair<float, int>>());

    if (scores_vec[0].first > pred_iou_thresh) {
        sam_result_t sam_result;
        ncnn::Mat mask = masks.channel(scores_vec[0].second);
        cv::Mat cv_mask_32f = cv::Mat::zeros(cv::Size(mask.w, mask.h), CV_32F);
        std::copy((float*)mask.data, (float*)mask.data + mask.w * mask.h, (float*)cv_mask_32f.data);
        
        cv::Mat single_mask_32f;
        cv::resize(cv_mask_32f(cv::Rect(0, 0, image_info.pad_w, image_info.pad_h)), single_mask_32f, cv::Size(image_info.img_w,image_info.img_h), 0, 0, 1);

        float stable_score = calculate_stability_score(single_mask_32f);
        if (stable_score < stability_score_thresh)
            return -1;

        single_mask_32f = single_mask_32f > 0;
        single_mask_32f.convertTo(sam_result.mask, CV_8UC1, 1, 0);
        
        if (postprocess_mask(sam_result.mask, sam_result.box) < 0)
            return -1;

        sam_results.push_back(sam_result);
    }
    else {
        return -1;
    }

    return 0;
}
int SegmentAnything::postprocess_mask(cv::Mat& mask, cv::Rect& box)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if(contours.size() == 0)
        return -1;

    if (contours.size() > 1) {
        float max_area = 0;
        int max_idx = 0;
        std::vector<std::pair<float,int>> areas;
        for (size_t i = 0; i < contours.size(); ++i) {
            float area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_idx = i;
                max_area = area;
            }
            areas.push_back(std::pair<float,int>(area,i));
        }
        
        for (size_t i = 0; i < areas.size(); ++i) {
            //if (i == max_idx)
            //    continue;
            //else {
            //    cv::drawContours(mask, contours, i, cv::Scalar(0), -1);
            //}
            if(areas[i].first < max_area * 0.3){
                cv::drawContours(mask, contours, i, cv::Scalar(0), -1);
            }
            else{
                box = box | cv::boundingRect(contours[i]);
            }
        }
    }
    else {
        box = cv::boundingRect(contours[0]);
    }
    return 0;
}
float SegmentAnything::calculate_stability_score(cv::Mat& mask, float mask_threshold, float stable_score_offset)
{
    float intersections = (float)cv::countNonZero(mask > (mask_threshold + stable_score_offset));
    float unions = (float)cv::countNonZero(mask > (mask_threshold - stable_score_offset));
    
    return intersections / unions;
}
}