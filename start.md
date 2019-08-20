<!-- 一 运行代码之前
1.安装dlib
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake ..
make -j8
make
make install
cp ./bin/* /usr/local/ -r

2.安装opencv 
github 下载 release 版本opencv 2.4.13 以上
编译 类似于dlib， 最后一样安装/usr/local 下
-->
一.生成人脸图像特征库
1.cd run
2.执行 ./install.sh  在test下生成可执行文件
3.运行 test 下的 ./test  传入单张图片,返回图片特征数据 
4.数据格式 128维特征向量 图片名字 （中间由空格分割） 

二.执行script 下的python文件
test.txt 是测试图片生成的128维特征向量（需要自己根据image图片 生成， 这里的test.txt是我随便放入的一个人脸 测试结果为编号2）

python script/face_recognition_knn.py data/database.txt data/test.txt  将测试图片与存储库中的图片对比
打印结果就是测试人脸对应的编号


