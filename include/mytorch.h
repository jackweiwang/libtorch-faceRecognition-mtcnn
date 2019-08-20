#ifndef MY_TORCH
#define MY_TORCH

#include <iostream>
#include <cstring>

#include "opencv2/opencv.hpp"


int init_torch(std::string model_path);

int get_torch_face_feature(cv::Mat frame, std::vector<float> &list);

#endif