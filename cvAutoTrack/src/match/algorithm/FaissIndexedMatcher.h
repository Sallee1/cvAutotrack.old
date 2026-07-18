#pragma once
#include "IIndexedMatchAlgorithm.h"
#include <functional>
#include <memory>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <faiss/IndexBinary.h>

namespace fs = std::filesystem;

/// Faiss 二进制索引匹配器（依赖注入 IndexBinary 工厂）
///
/// 用法：
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::ivf(256));
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hnsw(64));
///   auto idx = std::make_shared<FaissIndexedMatcher>(faiss_factory::hash(4, 16));
class FaissIndexedMatcher : public IIndexedMatchAlgorithm {
public:
	using FactoryFunc = std::function<std::unique_ptr<faiss::IndexBinary>(int d, int nb)>;
	/// @param factory 索引工厂: (int d, int nb) → faiss::IndexBinary*
	///               nb 为训练向量数，工厂可用 nb 自动推算适合的参数。
	explicit FaissIndexedMatcher(FactoryFunc factory)
		: m_factory(std::move(factory)) {}
	~FaissIndexedMatcher() override;

	void build(const cv::Mat& train_descriptors) override;
	bool try_load(const fs::path& path, const cv::Mat& train_descriptors) override;
	bool save(const fs::path& path) override;
	std::vector<std::vector<cv::DMatch>> knnmatch(const cv::Mat& query_descriptors, int k = 2) override;
	std::vector<cv::DMatch> match(const cv::Mat& query_descriptors) override;
	bool empty() const override;

private:
	std::unique_ptr<faiss::IndexBinary> m_index = nullptr;
	cv::Mat m_train_descriptors;
	FactoryFunc m_factory;
};

/// 预定义 Faiss 索引工厂
///
/// 所有工厂签名为 (int d, int nb)：
///   d  — 描述子位数（bit）
///   nb — 训练向量数（可由 FaissIndexedMatcher::build 自动传入）
namespace faiss_factory {

	/// 倒排索引（IndexBinaryIVF）
	/// @param nlist 聚类中心数。0 = auto sqrt(nb)，建议保留默认。
	///              可手工指定，例如 256~1024。
	FaissIndexedMatcher::FactoryFunc ivf(int nlist = 0);

	/// 图索引（IndexBinaryHNSW）
	/// @param M 连接数，越大召回越高（默认 32）
	FaissIndexedMatcher::FactoryFunc hnsw(int M = 32);

	/// 多索引哈希（IndexBinaryMultiHash）
	/// @param nhash 哈希表数量（推荐 2~4）
	/// @param b 每表位数。0 = auto = d / nhash，建议保留默认。
	FaissIndexedMatcher::FactoryFunc hash(int nhash = 0, int b = 0);

} // namespace faiss_factory
