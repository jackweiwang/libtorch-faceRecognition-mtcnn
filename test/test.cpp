#include <iostream>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <cstddef>
#include <cstring>
#include <time.h>
using namespace std;

typedef struct FaceInfo
{
    char* id;
    int rect[4];
    float face_data[512];
}FACE_INFO;


typedef int (*SET_P_PARAME)(const char *imagedir, FACE_INFO* faceInfo, int getface, int *sendface);
typedef int (*SET_P)(FACE_INFO* faceInfo, FACE_INFO* faceSet, int length, int *index, float *retMaxScore);
#define SINGLE_IMAGE 1

/*

ret_single_image_human_face_descriptor 返回单张图片的特征向量


*/
int main(int argc, char* argv[])
{
    cout << "start process." << endl;
    //加载so
    void *handle = dlopen("../build/libfaceRecognition.so", RTLD_LAZY);
    if(handle == NULL)
        cout << dlerror() <<endl;
    else
        cout << "dlopen so success" << endl;
    SET_P_PARAME single_face=NULL;
    *(void **)(&single_face) = dlsym(handle,"detect_image_face_info");
    if(single_face == NULL)
        cout << dlerror() <<endl;
    else
        cout << "dlsym so success" << endl;

    //调用so内接口
    string imagedir = argv[1];
    string imagedir1 = argv[2];
    const char *image = imagedir.c_str();
    const char *image1 = imagedir1.c_str();
    int setfacenum = 1;
    int getfacenum = 0; 
    FACE_INFO face[1];

    single_face(image, face, 1, &getfacenum);


    FACE_INFO face1[1];
    single_face(image1, face1, 1, &getfacenum);


    cout<< "getfacenum: " << getfacenum << endl;
    

    //test 2 ---------------------------------------------
    SET_P faceinfo=NULL;
    *(void **)(&faceinfo) = dlsym(handle,"search_face");
    if(faceinfo == NULL)
        cout << dlerror() <<endl;
    else
        cout << "dlsym so success" << endl;


    int id = 10;
    float retScore = 0.0;
    long beginTime =clock();//获得开始时间，单位为毫秒
    faceinfo(face, face1, 1, &id, &retScore);
    long endTime=clock();//获得结束时间
    cout<<"calculate_score_time:"<<endTime-beginTime<<endl;

    dlclose(handle);
    return 0;
}
