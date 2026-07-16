#include "pch.h"
#include "FaissIndexedMatcher.h"

#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/index_io.h>

// ===== FaissIndexedMatcher =====

FaissIndexedMatcher::~FaissIndexedMatcher()
{
	std::lock_guard<std::mutex> lock(m_mutex);
	delete m_index;
}

static bool cache_hit(faiss::IndexBinary* idx, const cv::Mat& cached, const cv::Mat& incoming)
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
	std::lock_guard<std::mutex> lock(m_mutex);
	if (cache_hit(m_index, m_train_descriptors, train_descriptors)) return;

	delete m_index;
	m_index = nullptr;

	int d = train_descriptors.cols * 8; // bits
	int nb = train_descriptors.rows;
	const auto* data = reinterpret_cast<const uint8_t*>(train_descriptors.data);

	auto* idx = m_factory(d);
	idx->train(nb, data);
	idx->add(nb, data);

	m_index = idx;
	m_train_descriptors = train_descriptors;
}

bool FaissIndexedMatcher::try_load(const fs::path& path, const cv::Mat& train_descriptors)
{
	if (train_descriptors.empty()) return false;
	std::lock_guard<std::mutex> lock(m_mutex);
	if (cache_hit(m_index, m_train_descriptors, train_descriptors)) return true;

	try {
		auto* idx = faiss::read_index_binary(path.u8string().c_str());
		if (idx) {
			delete m_index;
			m_index = idx;
			m_train_descriptors = train_descriptors;
			return true;
		}
	} catch (...) {}
	return false;
}

bool FaissIndexedMatcher::save(const fs::path& path)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	if (!m_index) return false;
	try {
		faiss::write_index_binary(m_index, path.u8string().c_str());
		return true;
	} catch (...) {}
	return false;
}

bool FaissIndexedMatcher::empty() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return !m_index || m_index->ntotal == 0;
}

std::vector<std::vector<cv::DMatch>> FaissIndexedMatcher::knnmatch(const cv::Mat& query_descriptors, int k)
{
	std::vector<std::vector<cv::DMatch>> result;
	if (query_descriptors.empty()) return result;

	faiss::IndexBinary* idx;
	{ std::lock_guard<std::mutex> lock(m_mutex); idx = m_index; }
	if (!idx || idx->ntotal == 0) return result;

	int nq = query_descriptors.rows;
	auto* q = reinterpret_cast<const uint8_t*>(query_descriptors.data);
	std::vector<faiss::idx_t> indices(nq * k);
	std::vector<int32_t> distances(nq * k);

	idx->search(nq, q, k, distances.data(), indices.data());

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

std::function<faiss::IndexBinary*(int d)> ivf(int nlist)
{
	return [nlist](int d) -> faiss::IndexBinary* {
		auto* quantizer = new faiss::IndexBinaryFlat(d);
		int nl = nlist > 0 ? nlist : 1;
		return new faiss::IndexBinaryIVF(quantizer, d, nl);
	};
}

std::function<faiss::IndexBinary*(int d)> hnsw(int M)
{
	return [M](int d) -> faiss::IndexBinary* {
		return new faiss::IndexBinaryHNSW(d, M);
	};
}

std::function<faiss::IndexBinary*(int d)> hash(int nhash, int b)
{
	return [nhash, b](int d) -> faiss::IndexBinary* {
		return new faiss::IndexBinaryMultiHash(d, nhash, b);
	};
}

} // namespace faiss_factory
