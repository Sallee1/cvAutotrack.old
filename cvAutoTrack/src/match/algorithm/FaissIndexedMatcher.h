#pragma once
#include "IIndexedMatchAlgorithm.h"
#include <functional>
#include <memory>
#include <mutex>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace faiss { class IndexBinary; }

namespace fs = std::filesystem;

/// Faiss 二进制索引匹配器（依赖注入 IndexBinary 工厂）
///
/// 用法：
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::ivf(256));
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hnsw(64));
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hash(4, 16));
class FaissIndexedMatcher : public IIndexedMatchAlgorithm {
public:
	/// @param factory 索引工厂: (int d) → faiss::IndexBinary*
	explicit FaissIndexedMatcher(std::function<faiss::IndexBinary*(int d)> factory)
		: m_factory(std::move(factory)) {}
	~FaissIndexedMatcher() override;

	void build(const cv::Mat& train_descriptors) override;
	bool try_load(const fs::path& path, const cv::Mat& train_descriptors) override;
	bool save(const fs::path& path) override;
	std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, int k = 2) override;
	std::vector<cv::DMatch> match(const cv::Mat& query_descriptors) override;
	bool empty() const override;

private:
	mutable std::mutex m_mutex;
	faiss::IndexBinary* m_index = nullptr;
	cv::Mat m_train_descriptors;
	std::function<faiss::IndexBinary*(int d)> m_factory;
};

/// 预定义 Faiss 索引工厂
namespace faiss_factory {

/// 倒排索引（IndexBinaryIVF）
/// @param nlist 聚类中心数，0 = auto sqrt(N)
std::function<faiss::IndexBinary*(int d)> ivf(int nlist = 0);

/// 图索引（IndexBinaryHNSW）
/// @param M 连接数，越大召回越高（默认 32）
std::function<faiss::IndexBinary*(int d)> hnsw(int M = 32);

/// 多索引哈希（IndexBinaryMultiHash）
/// @param nhash 哈希表数量
/// @param b 每表位数，0 = auto
std::function<faiss::IndexBinary*(int d)> hash(int nhash = 2, int b = 0);

} // namespace faiss_factory
