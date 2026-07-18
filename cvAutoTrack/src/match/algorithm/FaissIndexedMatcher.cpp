#include "pch.h"
#include "FaissIndexedMatcher.h"

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/index_io.h>

// ===== FaissIndexedMatcher =====

FaissIndexedMatcher::~FaissIndexedMatcher()
{
}

static bool cache_hit(const std::unique_ptr<faiss::IndexBinary>& idx, const cv::Mat& cached, const cv::Mat& incoming)
{
	return (idx != nullptr)
		&& cached.data == incoming.data
		&& cached.rows == incoming.rows
		&& cached.cols == incoming.cols
		&& cached.type() == incoming.type();
}

void FaissIndexedMatcher::build(const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return;
	if (cache_hit(m_index, m_train_descriptors, train_descriptors)) return;

	m_index = nullptr;

	int d = train_descriptors.cols * 8; // bits
	int nb = train_descriptors.rows;
	const auto* data = reinterpret_cast<const uint8_t*>(train_descriptors.data);

	std::unique_ptr<faiss::IndexBinary> idx = std::move(m_factory(d, nb));
#ifdef _CVAT_DEBUG_LOG
    auto __begin_time = std::chrono::steady_clock::now();
#endif
	idx->train(nb, data);
	idx->add(nb, data);
#ifdef _CVAT_DEBUG_LOG
    auto __build_index_time_len = std::chrono::steady_clock::now() - __begin_time;
    printf("[DEBUG] 索引构建耗时：%lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(__build_index_time_len).count());
#endif

	m_index = std::move(idx);
	m_train_descriptors = train_descriptors;
}

bool FaissIndexedMatcher::try_load(const fs::path& path, const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return false;
	if (cache_hit(m_index, m_train_descriptors, train_descriptors)) return true;

	try {
		auto idx = std::unique_ptr<faiss::IndexBinary>(faiss::read_index_binary(path.u8string().c_str()));
		if (idx) {
			m_index = std::move(idx);
			m_train_descriptors = train_descriptors;
			return true;
		}
	} catch (...) {}
	return false;
}

bool FaissIndexedMatcher::save(const fs::path& path)
{
	if (!m_index) return false;
	try {
		faiss::write_index_binary(m_index.get(), path.u8string().c_str());
		return true;
	} catch (...) {}
	return false;
}

bool FaissIndexedMatcher::empty() const
{
	return !m_index || m_index->ntotal == 0;
}

std::vector<std::vector<cv::DMatch>> FaissIndexedMatcher::knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> result;
	if (query_descriptors.empty()) return result;

	if (!m_index || m_index->ntotal == 0) return result;

	int nq = query_descriptors.rows;
	auto* q = reinterpret_cast<const uint8_t*>(query_descriptors.data);
	std::vector<faiss::idx_t> indices(nq * k);
	std::vector<int32_t> distances(nq * k);

	m_index->search(nq, q, k, distances.data(), indices.data());

	result.resize(nq);
	for (int i = 0; i < nq; ++i) {
		result[i].reserve(k);
		for (int j = 0; j < k; ++j) {
			auto tidx = indices[i * k + j];
			if (tidx >= 0)
				result[i].emplace_back(i, static_cast<int>(tidx), static_cast<float>(distances[i * k + j]));
		}
	}
	return result;
}

std::vector<cv::DMatch> FaissIndexedMatcher::match(const cv::Mat& query_descriptors)
{
	auto knn = knnmatch(query_descriptors, 1);
	std::vector<cv::DMatch> result;
	result.reserve(knn.size());
	for (auto& row : knn)
		if (!row.empty()) result.push_back(row[0]);
	return result;
}

// ===== 工厂函数 =====

namespace faiss_factory {
	FaissIndexedMatcher::FactoryFunc ivf(int nlist)
	{
		return [nlist](int d, int nb) ->std::unique_ptr<faiss::IndexBinary>
		{
			int nl = nlist > 0 ? nlist : static_cast<int>(std::sqrt(nb));
            auto* floatIndex = new faiss::IndexBinaryFlat(d);
			auto idx = std::make_unique<faiss::IndexBinaryIVF>(floatIndex, d, nl);
			idx->own_fields = true;
			return idx;
		};
	}

	FaissIndexedMatcher::FactoryFunc hnsw(int M)
	{
		return [M](int d, int /*nb*/) -> std::unique_ptr<faiss::IndexBinary> {
			auto idx = std::make_unique<faiss::IndexBinaryHNSW>(d, M);
			idx->own_fields = true;
			return idx;
		};
	}

	FaissIndexedMatcher::FactoryFunc hash(int nhash, int b)
	{
		return [nhash, b](int d, int nb) -> std::unique_ptr<faiss::IndexBinary> {
			int _b = b == 0 ? 32 : b;
			int _nhash = nhash == 0 ? d / _b : nhash;
			return std::make_unique<faiss::IndexBinaryMultiHash>(d, _nhash, _b);
		};
	}

} // namespace faiss_factory
