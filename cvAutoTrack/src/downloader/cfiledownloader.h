#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <openssl/md5.h>
#include <openssl/evp.h>
#include <curl/cURL.h>

namespace tianli {
	class FileDownloader {
	public:
		FileDownloader(const std::string& filePath,
			const std::string& url,
			const std::string& md5 = "")
			: m_file_path(filePath), m_url(url), m_md5(md5) {
		}

		bool download() {
			// 检查MD5匹配情况
			if (!m_md5.empty() && fileExists() && checkMD5()) {
				last_error_code = 0;
				m_last_error_msg = "File already exists and MD5 matches";
				return true;
			}

			// 执行下载
			bool download_success = downloadFile();

			// 需要MD5校验但下载失败
			if (!m_md5.empty() && !download_success) {
				return false;
			}

			// 需要MD5校验且下载成功
			if (!m_md5.empty()) {
				return checkMD5();
			}

			// 不需要MD5校验
			return download_success;
		}

		int getLastErrorCode() const { return last_error_code; }
		std::string getLastErrorMsg() const { return m_last_error_msg; }

		/**
		 * @brief 测试服务器连通性（HEAD 请求）
		 * @param url 测试地址
		 * @param timeout_sec 超时秒数（默认 10）
		 * @return 服务器正常返回 true
		 * @throw std::runtime_error 连接失败或 HTTP 错误时抛出
		 */
		static bool testConnection(const std::string& url, long timeout_sec = 10) {
			CURL* curl = curl_easy_init();
			if (!curl)
				throw std::runtime_error("testConnection: 初始化 CURL 失败");

			curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
			curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
			curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
			curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

			CURLcode res = curl_easy_perform(curl);
			if (res != CURLE_OK) {
				std::string err = curl_easy_strerror(res);
				curl_easy_cleanup(curl);
				throw std::runtime_error("服务器连接失败: " + err);
			}

			long http_code = 0;
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
			curl_easy_cleanup(curl);

			if (http_code >= 400)
				throw std::runtime_error("服务器返回错误: HTTP " + std::to_string(http_code));

			return true;
		}

	private:
		int last_error_code{ 0 };
		std::string m_last_error_msg;
		std::string m_file_path;
		std::string m_url;
		std::string m_md5;

		// 检查文件是否存在
		bool fileExists() const {
			std::ifstream file(m_file_path, std::ios::binary);
			return file.good();
		}

		// 计算文件MD5
		std::string calculateMD5() const {
			std::ifstream file(m_file_path, std::ios::binary);
			if (!file) return "";

#if OPENSSL_VERSION_MAJOR >= 3
			// OpenSSL 3.0及以上推荐使用EVP接口
			EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
			if (!mdctx) return "";

			const EVP_MD* md = EVP_md5();
			if (!md) {
				EVP_MD_CTX_free(mdctx);
				return "";
			}

			if (EVP_DigestInit_ex(mdctx, md, nullptr) != 1) {
				EVP_MD_CTX_free(mdctx);
				return "";
			}

			char buffer[4096];
			while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
				if (EVP_DigestUpdate(mdctx, buffer, file.gcount()) != 1) {
					EVP_MD_CTX_free(mdctx);
					return "";
				}
			}

			unsigned char digest[MD5_DIGEST_LENGTH];
			unsigned int digest_len = 0;
			if (EVP_DigestFinal_ex(mdctx, digest, &digest_len) != 1) {
				EVP_MD_CTX_free(mdctx);
				return "";
			}
			EVP_MD_CTX_free(mdctx);

			char md5_str[33]{};
			for (unsigned int i = 0; i < digest_len; ++i) {
				sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
			}
			return std::string(md5_str, 32);
#else
			MD5_CTX context;
			MD5_Init(&context);

			char buffer[4096];
			while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
				MD5_Update(&context, buffer, file.gcount());
			}

			unsigned char digest[MD5_DIGEST_LENGTH];
			MD5_Final(digest, &context);

			char md5_str[33]{};
			for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
				sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
			}
			return std::string(md5_str, 32);
#endif
		}

		// 校验MD5（忽略大小写，兼容服务器大写格式）
		bool checkMD5() const {
			if (m_md5.empty()) return true; // 未提供MD5时不校验
			std::string file_md5 = calculateMD5();
			if (file_md5.empty()) return false;
			if (file_md5.size() != m_md5.size()) return false;
			return _stricmp(file_md5.c_str(), m_md5.c_str()) == 0;
		}

		// CURL回调函数
		static size_t writeData(void* ptr, size_t size, size_t nmemb, void* stream) {
			std::ofstream* file = static_cast<std::ofstream*>(stream);
			size_t totalSize = size * nmemb;
			if (file && file->write(static_cast<char*>(ptr), totalSize)) {
				return totalSize; // 返回实际写入的字节数
			}
			return 0; // 写入失败
		}

		// 下载文件核心函数
		bool downloadFile() {
			CURL* curl = curl_easy_init();
			if (!curl) {
				last_error_code = -1;
				m_last_error_msg = "CURL initialization failed";
				return false;
			}

			std::ofstream output_file(m_file_path, std::ios::binary);
			std::cout << "Downloading from URL: " << m_url << " to file: " << m_file_path << std::endl;
			if (!output_file.is_open()) {
				last_error_code = -2;
				m_last_error_msg = "Failed to open file: " + m_file_path;
				curl_easy_cleanup(curl);
				return false;
			}

			// TODO: 临时措施，服务器可能转发到无证书的CDN
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);

			// 设置CURL选项
			curl_easy_setopt(curl, CURLOPT_URL, m_url.c_str());
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeData);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_file);
			curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
			curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
			curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

			// 执行下载
			CURLcode res = curl_easy_perform(curl);
			output_file.close();

			// 处理结果
			if (res != CURLE_OK) {
				last_error_code = res;
				m_last_error_msg = curl_easy_strerror(res);

				// 获取重定向URL（如果有）
				char* redirectUrl = nullptr;
				if (curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &redirectUrl) == CURLE_OK && redirectUrl) {
					m_last_error_msg += " (Redirected to: " + std::string(redirectUrl) + ")";
				}

				remove(m_file_path.c_str());
				curl_easy_cleanup(curl);
				return false;
			}

			// 验证下载是否真的成功
			long http_code = 0;
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
			if (http_code >= 400) {
				last_error_code = http_code;
				m_last_error_msg = "HTTP error: " + std::to_string(http_code);
				remove(m_file_path.c_str());
				curl_easy_cleanup(curl);
				return false;
			}

			curl_easy_cleanup(curl);
			last_error_code = 0;
			m_last_error_msg = "Download succeeded";
			return true;
		}
	};
}