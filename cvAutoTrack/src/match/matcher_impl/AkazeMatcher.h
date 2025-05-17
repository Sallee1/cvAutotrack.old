#pragma once
#include <match/IMatcher.h>

class AKAZEMatcher :public IMatcher {
private:
    cv::Ptr<cv::AKAZE> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    bool is_binary_descriptor = false;
    bool is_bf_matcher = true;

public:
    AKAZEMatcher(cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB,
        int descriptor_size = 0, int descriptor_channels = 3,
        float threshold = 0.001f, int nOctaves = 4,
        int nOctaveLayers = 4, cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2,
        int max_points = -1)
    {
        detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity, max_points);
        if (descriptor_type == cv::AKAZE::DESCRIPTOR_MLDB || descriptor_type == cv::AKAZE::DESCRIPTOR_MLDB_UPRIGHT)
        {
            is_binary_descriptor = true;
        }
        else {
            is_binary_descriptor = false;
        }
    }
    virtual ~AKAZEMatcher() = default;

    // 通过 IMatcher 继承
    std::vector<std::vector<cv::DMatch>> match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool bfmatch) override {
        std::vector<std::vector<cv::DMatch>> match_group;
        if (!is_bf_matcher && bfmatch)
        {
            if (is_binary_descriptor == true)
            {
                matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
            }
            else {
                matcher = cv::BFMatcher::create(cv::NORM_L2);
            }
            is_bf_matcher = true;
        }
        else if (is_bf_matcher && !bfmatch)
        {
            if (is_binary_descriptor == true)
            {
                cv::Ptr<cv::flann::LshIndexParams> index_params{ cv::makePtr<cv::flann::LshIndexParams>(6, 32, 2) };
                cv::Ptr<cv::flann::SearchParams> search_params{ cv::makePtr<cv::flann::SearchParams>(200, 0.1, true) };
                matcher = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
            }
            else
            {
                matcher = cv::makePtr<cv::FlannBasedMatcher>();     //默认KD树索引
            }

            is_bf_matcher = false;
        }
        matcher->knnMatch(query_descriptors, train_descriptors, match_group, 2);
        return match_group;
    }

    bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override {
        if (img.empty()) return  false;
        detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        if (keypoints.size() == 0) return false;
        return true;
    }
};