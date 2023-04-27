#include "segment_anything.h"
SegmentAnything::~SegmentAnything()
{
	image_encoder_net_.clear();
	mask_decoder_net_.clear();
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

	const float meanVals[3] = { 123.675f, 116.28f,  103.53f };
	const float normVals[3] = { 0.01712475f, 0.0175f, 0.01742919f };
	in_pad.substract_mean_normalize(meanVals, normVals);

	ncnn::Extractor image_encoder_ex = image_encoder_net_.create_extractor();

	image_encoder_ex.input("image", in_pad);
	image_encoder_ex.extract("image_embeddings", image_embeddings);

	image_info.img_h = img_h;
	image_info.img_w = img_w;
	image_info.pad_h = h;
	image_info.pad_w = w;


	return 0;
}

int SegmentAnything::embed_masks(prompt_info_t prompt_info, ncnn::Mat& mask_input, ncnn::Mat& has_mask)
{
	mask_input = ncnn::Mat(256, 256, 1);
	mask_input.fill(0.f);
	has_mask = ncnn::Mat(1);
	has_mask.fill(0.f);

	return 0;
}

int SegmentAnything::embed_points(prompt_info_t prompt_info, std::vector<ncnn::Mat>& point_labels, ncnn::Mat& point_coords)
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
int SegmentAnything::MaskDecoder(const ncnn::Mat& image_embeddings, image_info_t& image_info, prompt_info_t prompt_info, cv::Mat& single_mask)
{
	std::vector<ncnn::Mat> point_labels;
	ncnn::Mat point_coords;
	embed_points(prompt_info, point_labels, point_coords);

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


	std::vector<std::pair<float, int>> socres_vec;
	for (int i = 0; i < scores.w; ++i) {
		socres_vec.push_back(std::pair<float, int>(scores[i], i));
	}
	std::sort(socres_vec.begin(), socres_vec.end(), std::greater<std::pair<float, int>>());

	ncnn::Mat mask = masks.channel(socres_vec[0].second);

	cv::Mat cv_mask_32f = cv::Mat::zeros(cv::Size(mask.w, mask.h), CV_32F);
	std::copy((float*)mask.data, (float*)mask.data + mask.w * mask.h, (float*)cv_mask_32f.data);

	cv_mask_32f = cv_mask_32f > 0;
	cv::Mat cv_mask;
	cv_mask_32f.convertTo(cv_mask, CV_8UC1, 1, 0);

	cv::resize(cv_mask(cv::Rect(0, 0, image_info.pad_w / 4, image_info.pad_h / 4)), single_mask, cv::Size(image_info.img_w,image_info.img_h), 0, 0, cv::INTER_AREA);

	return 0;
}