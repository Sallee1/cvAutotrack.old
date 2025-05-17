#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class IMatcher {
public:

    virtual ~IMatcher() = default;
    struct KeyMatPoint
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        bool empty() { return keypoints.size() == 0; }
        auto size() { return keypoints.size(); }
    };

public:
    virtual std::vector<std::vector<cv::DMatch>> match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool bfmatch = false) = 0;

    virtual std::vector<std::vector<cv::DMatch>> match(const KeyMatPoint& query_key_mat_point, const KeyMatPoint& train_key_mat_point, bool bfmatch = false) {
        return match(query_key_mat_point.descriptors, train_key_mat_point.descriptors, bfmatch);
    }

    virtual bool detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) = 0;

    virtual bool detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point) {
        return detect_and_compute(img, key_mat_point.keypoints, key_mat_point.descriptors);
    }
};