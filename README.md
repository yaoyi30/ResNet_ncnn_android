# ResNet_ncnn_android
This is an android app about monkey image classification

dataset used for training:https://www.kaggle.com/slothkong/10-monkey-species

ResNet training code:https://github.com/yaoyi30/ResNet_Image_Classification_PyTorch

# how to build and run
## step1
https://github.com/Tencent/ncnn/releases

1.Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself

2.Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

## step2
Open this project with Android Studio, build it and enjoy!

# screenshot

![d9c3345023acaf03e7baee8a48f9337](https://user-images.githubusercontent.com/56180347/174019245-ad39a48f-e2d7-4be6-98b5-6a55cf8d4481.jpg)

# Reference

https://github.com/nihui/ncnn-android-squeezenet
