#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <spdhelper.hpp>
#include <BTimer.hpp>
#include "MTCNN.h"
#include "mytorch.h"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include "dlib_face_recognition.h"
#include "dlib/matrix.h"
#include "dlib/opencv.h"
using namespace std;
typedef struct FaceInfo
{
    char *id;
    int rect[4];
    float face_data[512];
}FACEINFO;

int get_files(string base_path, std::vector<char *> &image_paths)
{
	DIR *dir;
	struct dirent *ptr;
	if ((dir = opendir(base_path.c_str())) == NULL)
	{
		printf("Open dir: %s Error...\n", base_path.c_str());
		exit(1);
	}
	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
			continue;
		if (ptr->d_type == 8)    ///file  (.jpg / .png)
		{
			//printf("d_name:%s/%s\n",base_path,ptr->d_name);
			/// do strings split joint
			if (strcmp(ptr->d_name + strlen(ptr->d_name) - 3, "jpg") && strcmp(ptr->d_name + strlen(ptr->d_name) - 3, "png")) continue;
			char *tmp = (char *)calloc(0, (strlen(base_path.c_str()) + strlen(ptr->d_name) + 5) * sizeof(char *));

			sprintf(tmp, "%s%c%s", base_path.c_str(), '/', ptr->d_name);

			image_paths.push_back(tmp);
                        free(tmp);
		}
}
	closedir(dir);
	return 0;
}
static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}
int init_parameter(TModelParam &modelParam)
{

    std::string pnet_weight_path = std::string(MODEL_PATH) + "pnet.pt";
    std::string rnet_weight_path = std::string(MODEL_PATH) + "rnet.pt";
    std::string onet_weight_path = std::string(MODEL_PATH) + "onet.pt";

    TAlgParam alg_param;
    alg_param.min_face = 40;
    alg_param.scale_factor = 0.79;
    alg_param.cls_thre[0] = 0.6;
    alg_param.cls_thre[1] = 0.7;
    alg_param.cls_thre[2] = 0.7;


    modelParam.alg_param = alg_param;
    modelParam.model_path = {pnet_weight_path, rnet_weight_path, onet_weight_path};
    modelParam.mean_value = {{127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}};
    modelParam.scale_factor = {1.0f, 1.0f, 1.0f};
    modelParam.gpu_id = 0;
    modelParam.device_type = torch::DeviceType::CUDA;

    return 0;
}
int detect_face(const char *files, FACEINFO *fp_out, int getface, int *sendface)
{
    MTCNN mt;
    TModelParam param;
    shape_predictor sp;
    deserialize("../model/shape_predictor_68_face_landmarks.dat") >> sp;
    std::vector<full_object_detection>  shapes;

    init_parameter(param);
    mt.InitDetector(&param);

    cv::Mat src = cv::imread(files);
    if(!src.data)
    {
        LOGE("cannot load image!");
        return -1;
    }
    std::vector<cv::Rect> outFaces;

    mt.DetectFace(src, outFaces);

    dlib::cv_image<dlib::bgr_pixel> gimg(src);
    matrix<rgb_pixel> img;

    assign_image(img, gimg);

    //for(auto& i : outFaces)
    //    cv::rectangle(src, i, {0,255,0}, 2);

    //cv::imshow("result", src);
    //cv::waitKey(0);
    for (int i=0; i < outFaces.size(); i++ )
    {
        dlib::rectangle rec = openCVRectToDlib(outFaces[i]);

        full_object_detection shape_point = sp(img, rec);

        shapes.push_back(shape_point);
    }

    if (shapes.size() == 0)
    {
        cout << "image" << " " << files << " " << " no face" << endl;
        return -1;
    }

    string model_path = "../model/my_model_152.pt";
    *sendface = shapes.size();

    init_torch(model_path);

    for(unsigned int i=0; i < shapes.size(); i++)
    {
        std::vector<float> feature(512);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shapes[i], 112, 0.25), face_chip);

        cv::Mat dest = dlib::toMat(face_chip).clone();
        cv::imwrite("image.jpg", dest);

        get_torch_face_feature(dest, feature);

        if( getface > (int)i)
        {
            fp_out[i].rect[0] = outFaces[i].x;
            fp_out[i].rect[1] = outFaces[i].y;
            fp_out[i].rect[2] = outFaces[i].width;
            fp_out[i].rect[3] = outFaces[i].height;

            for (unsigned int  j = 0; j < 512; j++)
            {
                fp_out[i].face_data[j] = feature[j];
            }

        }

        feature.clear();
    }

    shapes.clear();
    //delete mt;
    return 0;

}
//余弦相似度
float ret_cos_face_score(float* feature1, float* feature2)
{
    float temp1, temp2, f11=0, f22=0, f12=0;
	// FEATURE_LEN = 512， 
	for(int i = 0; i < 512; i++)
	{
        temp1 = feature1[i];
		f11 += temp1 * temp1;
		temp2 = feature2[i];
		f22 += temp2 * temp2;
		f12 += temp1 * temp2;
	}

	float score = f12 / sqrtf(f11*f22);
	if(score < -1.f || score > 1.f)
	{
	    printf("score overflow:%f\n", score);
    }
	else if(score < 0.f)
    {
	    score = 0.f;
    }
	return score;
}
//////欧几里得距离
//float ret_face_score(float *x, float *y)
//{
//    float dist = 0.0;
//    for(unsigned int i=0; i < 512; i++)
//    {
//        dist += pow(x[i]-y[i], 2);
//    }
//
//    return sqrtf(dist);
//}

extern "C" {

int search_face(FACEINFO *srcface , FACEINFO *faceSet, int length, int *index, float *retMaxScore)
{
    float Score = 0.0;
    std::vector<float> data;
    for(int i=0; i < length; i++)
    {
        Score = ret_cos_face_score(srcface->face_data, faceSet[i].face_data);
        data.push_back(Score);
    }

    float threshold = 0.40;
    std::vector<float>::iterator  maxScore = std::max_element(std::begin(data), std::end(data));
    
    if( *maxScore > threshold )
    {
        *index = std::distance(std::begin(data), maxScore);
        *retMaxScore = *maxScore;
        printf("maxScore = %f  find person=%d\n", *maxScore, *index);
    }
    else
    {
        printf("Did not find this person! score=%f \n", *maxScore);
        return -1;
    }
    return 0;
}

int detect_image_face_info(const char *fileName, FaceInfo *faceInfo, int getface,  int *sendface)
{
    int result = 0;
    if (fileName != NULL)
    {
        result = detect_face(fileName, faceInfo, getface, sendface);
    }

    return result;
}

}
