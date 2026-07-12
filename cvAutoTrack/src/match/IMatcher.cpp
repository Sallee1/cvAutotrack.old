#include "pch.h"
#include "IMatcher.h"
#include "FlannIndex.h"

void IMatcher::cache_flann_train_descriptors(const cv::Mat& train_descriptors)
{
	if (m_flann_index)
		m_flann_index->build(train_descriptors);
}

bool IMatcher::try_load_flann_index(const std::string& path, const cv::Mat& train_descriptors)
{
	if (!m_flann_index) return false;
	return m_flann_index->try_load(path, train_descriptors);
}

bool IMatcher::save_flann_index(const std::string& path)
{
	if (!m_flann_index) return false;
	return m_flann_index->save(path);
}

std::vector<std::vector<cv::DMatch>> IMatcher::flann_knnmatch(const cv::Mat& query_descriptors, int k)
{
	if (!m_flann_index) return {};
	return m_flann_index->knnmatch(query_descriptors, k);
}

std::vector<std::vector<cv::DMatch>> IMatcher::flann_knnmatch(const KeyMatPoint& query, int k)
{
	return flann_knnmatch(query.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::flann_match(const cv::Mat& query_descriptors)
{
	if (!m_flann_index) return {};
	return m_flann_index->match(query_descriptors);
}

std::vector<cv::DMatch> IMatcher::flann_match(const KeyMatPoint& query)
{
	return flann_match(query.descriptors);
}

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> match_group;
	if (query_descriptors.empty() || train_descriptors.empty())
	{
		return match_group;
	}

	auto matcher = create_bf_matcher(false);
	matcher->knnMatch(query_descriptors, train_descriptors, match_group, k);
	return match_group;
}

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k)
{
	return bf_knnmatch(query.descriptors, train.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, bool cross_check)
{
	std::vector<cv::DMatch> matches;
	if (query_descriptors.empty() || train_descriptors.empty())
	{
		return matches;
	}

	auto matcher = create_bf_matcher(cross_check);
	matcher->match(query_descriptors, train_descriptors, matches);
	return matches;
}

std::vector<cv::DMatch> IMatcher::bf_match(const KeyMatPoint& query, const KeyMatPoint& train, bool cross_check)
{
	return bf_match(query.descriptors, train.descriptors, cross_check);
}

bool IMatcher::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	if (img.empty()) return false;
	return detect_impl(img, keypoints);
}

bool IMatcher::compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty() || keypoints.empty()) return false;
	return compute_impl(img, keypoints, descriptors);
}

bool IMatcher::detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty()) return false;

	// 确定要遍历的配置：金字塔尺度 × 亮度增益
	const auto& scales = m_pyramid_scales;
	const auto& gains = m_brightness_gains;
	bool simple = scales.empty() && gains.empty();

	if (simple)
		return detect_and_compute_impl(img, keypoints, descriptors);

	// 至少有一个维度展开：构造配置列表
	struct Config { double scale; double gain; };
	std::vector<Config> configs;
	if (scales.empty())
		configs.push_back({ 1.0, 1.0 });  // 仅多亮度时用原图亮度1.0做一次兜底
	for (double s : scales.empty() ? std::vector<double>{1.0} : scales)
	{
		for (double g : gains.empty() ? std::vector<double>{1.0} : gains)
		{
			configs.push_back({ s, g });
		}
	}

	keypoints.clear();

	// 线程局部 buffer，每个 config 独占一个 slot
	struct TileResult {
		bool ok = false;
		std::vector<cv::KeyPoint> kp;
		cv::Mat desc;
	};
	std::vector<TileResult> results(configs.size());

	cv::parallel_for_({ 0, (int)configs.size() }, [&](const cv::Range& range)
	{
		for (int i = range.start; i < range.end; i++)
		{
			const auto& cfg = configs[i];
			cv::Mat work = img.clone();

			// 缩放
			if (cfg.scale != 1.0)
			{
				cv::resize(work, work, cv::Size(), cfg.scale, cfg.scale, cv::INTER_AREA);
				if (work.cols < 16 || work.rows < 16) continue;
			}

			// 提亮
			if (cfg.gain != 1.0)
				work.convertTo(work, -1, cfg.gain, 0);

			auto& r = results[i];
			if (!detect_and_compute_impl(work, r.kp, r.desc)) continue;
			if (r.kp.empty()) continue;

			// 坐标还原
			if (cfg.scale != 1.0)
			{
				float inv = 1.0f / (float)cfg.scale;
				for (auto& p : r.kp)
				{
					p.pt.x *= inv;
					p.pt.y *= inv;
					p.size *= inv;
				}
			}
			r.ok = true;
		}
	});

	// 第一遍：统计总量 + 确定描述子格式
	int total_kp = 0;
	int desc_cols = 0, desc_type = -1;
	for (auto& r : results)
	{
		if (!r.ok) continue;
		total_kp += (int)r.kp.size();
		if (desc_cols == 0 && !r.desc.empty())
		{
			desc_cols = r.desc.cols;
			desc_type = r.desc.type();
		}
	}
	if (total_kp == 0) return false;

	// 预分配输出容器
	keypoints.clear();
	keypoints.reserve(total_kp);
	if (total_kp > 0)
		descriptors.create(total_kp, desc_cols, desc_type);

	// 第二遍：按块拷贝到预分配的描述子矩阵
	int offset = 0;
	for (auto& r : results)
	{
		if (!r.ok) continue;
		int n = (int)r.kp.size();
		r.desc.copyTo(descriptors.rowRange(offset, offset + n));
		keypoints.insert(keypoints.end(),
			std::make_move_iterator(r.kp.begin()),
			std::make_move_iterator(r.kp.end()));
		offset += n;
	}
	return true;
}

// --- 默认 _impl 实现（子类可覆写）---

bool IMatcher::detect_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	auto detector = getFeature2D();
	detector->detect(img, keypoints);
	return !keypoints.empty();
}

bool IMatcher::compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (keypoints.empty()) return false;
	auto detector = getFeature2D();
	detector->compute(img, keypoints, descriptors);
	return !descriptors.empty();
}

bool IMatcher::detect_and_compute_impl(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (!detect_impl(img, keypoints)) return false;
	if (keypoints.empty()) return false;
	return compute_impl(img, keypoints, descriptors);
}

bool IMatcher::detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point) {
	return detect_and_compute(img, key_mat_point.keypoints, key_mat_point.descriptors);
}

cv::Ptr<cv::DescriptorMatcher> IMatcher::create_bf_matcher(bool cross_check)
{
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (getIsBinaryDescriptor())
	{
		matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_check);
	}
	else
	{
		matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
	}
	return matcher;
}


