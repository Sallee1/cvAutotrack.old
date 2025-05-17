#pragma once
#include <match/IMatcher.h>
#include <opencv2/xfeatures2d.hpp>

class SURFMatcher :public IMatcher {
private:
    cv::Ptr<cv::xfeatures2d::SURF> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    bool is_bf_matcher = true;

public:
    SURFMatcher(double hessianThreshold = 100,
        int nOctaves = 4, int nOctaveLayers = 3,
        bool extended = false, bool upright = false)
    {
        detector = cv::xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    }
    virtual ~SURFMatcher() = default;

    // 通过 IMatcher 继承
    std::vector<std::vector<cv::DMatch>> match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool bfmatch) override {
        std::vector<std::vector<cv::DMatch>> match_group;
        if (!is_bf_matcher && bfmatch)
        {
            matcher = cv::BFMatcher::create(cv::NORM_L2);
            is_bf_matcher = true;
        }
        else if (is_bf_matcher && !bfmatch)
        {
            matcher = cv::makePtr<cv::FlannBasedMatcher>();
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