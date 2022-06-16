// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cmath>
#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

#include "res.id.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net squeezenet;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    squeezenet.opt = opt;

    // init param
    {
        int ret = squeezenet.load_param_bin(mgr, "res.param.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = squeezenet.load_model(mgr, "res.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
//    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, 224, 224);
    // squeezenet
    std::vector<float> cls_scores;
    {
        const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
        const float std_vals[3] = {1/58.395f, 1/57.12f, 1/57.375f};

        in.substract_mean_normalize(mean_vals, std_vals);

        ncnn::Extractor ex = squeezenet.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        ex.input(res_param_id::LAYER_input, in);

        ncnn::Mat out;
        ex.extract(res_param_id::BLOB_output, out);



        cls_scores.resize(out.w);
        for (int j=0; j<out.w; j++)
        {
            cls_scores[j] = out[j];
        }
    }

    static const char* class_names[] = {
            "n0","n1","n2","n3","n4","n5","n6","n7","n8","n9"
    };

    float sum = 0.0f;
    for (int j = 0; j < cls_scores.size(); j++)
    {
        float s_l = cls_scores[j];
        cls_scores[j] = std::exp(s_l);
        //scores[i] = std::exp(scores[i]);
        sum += cls_scores[j];
    }

    for (int a = 0; a < cls_scores.size(); a++)
    {
        cls_scores[a] /= sum;
    }


    int size = cls_scores.size();
    std::vector<std::pair<float,std::string>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i],class_names[i]);

    }


    std::sort(vec.begin(), vec.end(),std::greater<std::pair<float,std::string>>());

    char tmp[32];
    sprintf(tmp, "%.1f", vec[0].first*100);

    double elasped = ncnn::get_current_time() - start_time;

    char time[32];
    sprintf(time, "%.1fms",elasped);

    std::string result_str = "类别:"+vec[0].second+"  "+"概率值:"+tmp+"%"+"  "+"时间:"+time;


    jstring result = env->NewStringUTF(result_str.c_str());

//
//    // return top class
//    int top_class = 0;
//    float max_score = 0.f;
//    for (size_t i=0; i<cls_scores.size(); i++)
//    {
//        float s = cls_scores[i];
//        if (s > max_score)
//        {
//            top_class = i;
//            max_score = s;
//        }
//    }
//
//    const std::string& word = squeezenet_words[top_class];
//    char tmp[32];
//    sprintf(tmp, "%.3f", max_score);
//    std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;
//
//    // +10 to skip leading n03179701
//    jstring result = env->NewStringUTF(result_str.c_str());
//
//    double elasped = ncnn::get_current_time() - start_time;
//    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   detect", elasped);

    return result;
}

}
