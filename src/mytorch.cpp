#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <memory>
#include <vector>
using namespace std;
torch::jit::script::Module module;
/*
int l2_norm(vector<float> norm, vector<float> &tsqrt)
    {
        float sum = 0.0;

        for(int i=0; i< 512; i++)
        {
            sum += pow(norm[i], 2);
        }

        for(int i=0; i< 512; i++)
        {
            tsqrt[i] = norm[i]/sum;
            //printf("--%lf", tsqrt[i]);
        }

        return 0;
    }

// resize并保持图像比例不变
cv::Mat resize_with_ratio(cv::Mat& img)
{
    cv::Mat temImage;
    int w = img.cols;
    int h = img.rows;

    float t = 1.;
    float len = t * std::max(w, h);
    int dst_w = 224, dst_h = 224;
    cv::Mat image = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(128,128,128));
    cv::Mat imageROI;
    if(len==w)
    {
        float ratio = (float)h/(float)w;
        cv::resize(img,temImage,cv::Size(224,224*ratio),0,0,cv::INTER_LINEAR);
        imageROI = image(cv::Rect(0, ((dst_h-224*ratio)/2), temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }
    else
    {
        float ratio = (float)w/(float)h;
        cv::resize(img,temImage,cv::Size(224*ratio,224),0,0,cv::INTER_LINEAR);
        imageROI = image(cv::Rect(((dst_w-224*ratio)/2), 0, temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }

    return image;
}
*/
void adaptiveLogarithmicMapping(const cv::Mat& img, cv::Mat &dst)
 {
     cv::Mat ldrDrago;
     img.convertTo(ldrDrago, CV_32FC3, 1.0f/255);
     cv::cvtColor(ldrDrago, ldrDrago, cv::COLOR_BGR2XYZ);
     cv::Ptr<cv::TonemapDrago> tonemapDrago = cv::createTonemapDrago(1.f, 1.f, 0.85f);
     tonemapDrago->process(ldrDrago, dst);
     cv::cvtColor(dst, dst, cv::COLOR_XYZ2BGR);
     dst.convertTo(dst, CV_8UC3, 255);
 }

int init_torch(std::string model_path)
{
    module = torch::jit::load(model_path);
    module.to(at::kCUDA);

    return 0;
}
int get_torch_face_feature(cv::Mat frame, std::vector<float> &list)
{
    //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
    //assert(module != nullptr);
    cv::Mat image = frame.clone();
    cv::Mat input;
    cv::Mat adapt;

    cv::Mat imageRGB[3];
	cv::split(image, imageRGB);
	for (int i = 0; i < 3; i++)
	{
		cv::equalizeHist(imageRGB[i], imageRGB[i]);
	}
	cv::merge(imageRGB, 3, image);



    adaptiveLogarithmicMapping(image, adapt);

    cv::cvtColor(adapt, input, cv::COLOR_BGR2RGB);

    //cv::imwrite("input.jpg",input);

    // 下方的代码即将图像转化为Tensor，随后导入模型进行预测
    torch::Tensor tensor_image = torch::from_blob(input.data, {1,input.rows, input.cols,3}, torch::kByte);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.to(torch::kCUDA);
    torch::Tensor result = module.forward({tensor_image}).toTensor();

    //std::vector<float> norm(512);

    for(int i=0; i < 512; i++)
    {
        list[i] = (result[0][i].item().toFloat());//+ fresult[0][i].item().toFloat());
    }

    //l2_norm(norm, list);
    return 0;
}