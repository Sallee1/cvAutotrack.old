#include "pch.h"
#include "IMatcher.h"
#include "utils/utils.progress.h"

// ===== 特征提取（单层） =====

bool IMatcher::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints)
{
	if (img.empty() || !m_detector) return false;
	m_detector->detect(img, keypoints);
	return !keypoints.empty();
}

bool IMatcher::compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty() || keypoints.empty() || !m_descriptor) return false;
	m_descriptor->compute(img, keypoints, descriptors);
	return !descriptors.empty();
}

bool IMatcher::detect_and_compute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty()) return false;
	if (!m_detector || !m_descriptor) return false;

	// 同一对象时优化为 detectAndCompute
	if (m_detector == m_descriptor)
	{
		m_detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
		return !keypoints.empty();
	}

	m_detector->detect(img, keypoints);
	if (keypoints.empty()) return false;
	m_descriptor->compute(img, keypoints, descriptors);
	return !descriptors.empty();
}

bool IMatcher::detect_and_compute(const cv::Mat& img, KeyMatPoint& key_mat_point)
{
	return detect_and_compute(img, key_mat_point.keypoints, key_mat_point.descriptors);
}

// ===== 增强多层检测+描述（金字塔多尺度 × 亮度增益并行处理） =====

bool IMatcher::detect_and_compute_ex(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	if (img.empty()) return false;

	const auto& scales = m_pyramid_scales;
	const auto& gains = m_brightness_gains;
	bool simple = scales.empty() && gains.empty();

	if (simple)
		return detect_and_compute(img, keypoints, descriptors);

	// 构造配置列表
	struct Config { double scale; double gain; };
	std::vector<Config> configs;
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
			if (!detect_and_compute(work, r.kp, r.desc)) continue;
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

bool IMatcher::detect_and_compute_ex(const cv::Mat& img, KeyMatPoint& key_mat_point)
{
	return detect_and_compute_ex(img, key_mat_point.keypoints, key_mat_point.descriptors);
}

// ===== 索引匹配（委托 m_indexed_matcher） =====

void IMatcher::cache_train_descriptors(const cv::Mat& train_descriptors)
{
	if (!m_indexed_matcher)
		return;

	// 索引构建可能耗时（~1min），显示不定进度条
	TianLi::Utils::Win32ProgressWindow progress;
	progress.create_marquee(L"cvAutoTrack", L"正在构建特征索引...（约1分钟）");
	m_indexed_matcher->build(train_descriptors);
	progress.close();
}

bool IMatcher::try_load_index(const fs::path& path, const cv::Mat& train_descriptors)
{
	if (!m_indexed_matcher) return false;
	return m_indexed_matcher->try_load(path, train_descriptors);
}

bool IMatcher::save_index(const fs::path& path)
{
	if (!m_indexed_matcher) return false;
	return m_indexed_matcher->save(path);
}

std::vector<std::vector<cv::DMatch>> IMatcher::indexed_knnmatch(const cv::Mat& query_descriptors, int k)
{
	if (!m_indexed_matcher) return {};
	return m_indexed_matcher->knnmatch(query_descriptors, k);
}

std::vector<std::vector<cv::DMatch>> IMatcher::indexed_knnmatch(const KeyMatPoint& query, int k)
{
	return indexed_knnmatch(query.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::indexed_match(const cv::Mat& query_descriptors)
{
	if (!m_indexed_matcher) return {};
	return m_indexed_matcher->match(query_descriptors);
}

std::vector<cv::DMatch> IMatcher::indexed_match(const KeyMatPoint& query)
{
	return indexed_match(query.descriptors);
}

// ===== 暴力匹配（委托 m_bf_matcher） =====

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors, int k)
{
	if (!m_bf_matcher) return {};
	return m_bf_matcher->knnmatch(query_descriptors, train_descriptors, k);
}

std::vector<std::vector<cv::DMatch>> IMatcher::bf_knnmatch(const KeyMatPoint& query, const KeyMatPoint& train, int k)
{
	return bf_knnmatch(query.descriptors, train.descriptors, k);
}

std::vector<cv::DMatch> IMatcher::bf_match(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors)
{
	if (!m_bf_matcher) return {};
	return m_bf_matcher->match(query_descriptors, train_descriptors);
}

std::vector<cv::DMatch> IMatcher::bf_match(const KeyMatPoint& query, const KeyMatPoint& train)
{
	return bf_match(query.descriptors, train.descriptors);
}



