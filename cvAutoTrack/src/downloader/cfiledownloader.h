#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <openssl/md5.h>
#include <openssl/evp.h>

#include <curl/curl.h>
#include <rpcdce.h>
namespace fs = std::filesystem;
namespace tianli {
	class FileDownloader {
	public:
		FileDownloader(const fs::path& filePath,
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
		std::string getLastRedirectUrl() const { return m_last_redirect_url; }

		static std::string calcFileMD5(const fs::path& filePath)
		{
			std::ifstream file(filePath, std::ios::binary);
			if (!file) return "";

		#if OPENSSL_VERSION_MAJOR >= 3
			EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
			if (!mdctx) return "";
			const EVP_MD* md = EVP_md5();
			if (!md) { EVP_MD_CTX_free(mdctx); return ""; }
			if (EVP_DigestInit_ex(mdctx, md, nullptr) != 1) { EVP_MD_CTX_free(mdctx); return ""; }

			char buffer[4096];
			while (file.read(buffer, sizeof(buffer)) || file.gcount())
				EVP_DigestUpdate(mdctx, buffer, file.gcount());

			unsigned char digest[MD5_DIGEST_LENGTH];
			unsigned int digest_len = 0;
			if (EVP_DigestFinal_ex(mdctx, digest, &digest_len) != 1) { EVP_MD_CTX_free(mdctx); return ""; }
			EVP_MD_CTX_free(mdctx);

			char md5_str[33]{};
			for (unsigned int i = 0; i < digest_len; ++i)
				sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
			return std::string(md5_str, 32);
		#else
			MD5_CTX context;
			MD5_Init(&context);
			char buffer[4096];
			while (file.read(buffer, sizeof(buffer)) || file.gcount())
				MD5_Update(&context, buffer, file.gcount());
			unsigned char digest[MD5_DIGEST_LENGTH];
			MD5_Final(digest, &context);
			char md5_str[33]{};
			for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
				sprintf_s(&md5_str[i * 2], 3, "%02x", digest[i]);
			return std::string(md5_str, 32);
		#endif
		}

	private:
		int last_error_code{ 0 };
		std::string m_last_error_msg;
		std::string m_last_redirect_url;
		fs::path m_file_path;
		std::string m_url;
		std::string m_md5;

		// 检查文件是否存在
		bool fileExists() const {
			return fs::exists(m_file_path);
		}

		// 计算文件MD5
		std::string calculateMD5() const {
			return calcFileMD5(m_file_path);
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

		// 下载文件核心函数（手动处理重定向，记录重定向链）
		bool downloadFile() {
			CURL* curl = curl_easy_init();
			if (!curl) {
				last_error_code = -1;
				m_last_error_msg = "CURL initialization failed";
				return false;
			}

			// TODO: 临时措施，服务器可能转发到无证书的CDN
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);

			// 设置CURL选项（通用部分）
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeData);
			curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
			curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0L);   // 手动检查HTTP状态码
			curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L); // 手动处理重定向

			m_last_redirect_url.clear();
			std::string current_url = appendSidParam(m_url);
			const int max_redirects = 10;
			CURLcode res = CURLE_OK;
			bool success = false;

			// 连接超时（最多等10s建立TCP/TLS连接）
			curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
			// 弱网检测：速度低于 1KB/s 持续超过 10s 则自动中止
			curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
			curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 10L);

			for (int redirect_count = 0; redirect_count <= max_redirects; redirect_count++)
			{
				// 每次重定向重新打开文件（truncate 模式清除上次重定向的响应体）
				std::ofstream output_file(m_file_path, std::ios::binary | std::ios::out | std::ios::trunc);
				if (!output_file.is_open()) {
					last_error_code = -2;
					m_last_error_msg = std::string("Failed to open file: ") + m_file_path.u8string();
					curl_easy_cleanup(curl);
					return false;
				}

				std::cout << "Downloading from URL: " << current_url << " to file: " << m_file_path.u8string() << std::endl;
				curl_easy_setopt(curl, CURLOPT_URL, current_url.c_str());
				curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_file);

				res = curl_easy_perform(curl);
				output_file.close();

				// CURL 级错误
				if (res != CURLE_OK) {
					last_error_code = res;
					if (res == CURLE_OPERATION_TIMEDOUT) {
						// 根据是否收到过数据判断是"下载过慢"还是"服务器无响应"
						curl_off_t bytes_downloaded = 0;
						curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD_T, &bytes_downloaded);
						if (bytes_downloaded > 0)
							m_last_error_msg = "Download aborted: network speed too low (< 1KB/s) for more than 10 seconds";
						else
							m_last_error_msg = "Download aborted: server not responding (timeout)";
					} else {
						m_last_error_msg = curl_easy_strerror(res);
					}
					appendRedirectInfo();
					fs::remove(m_file_path);
					curl_easy_cleanup(curl);
					return false;
				}

				long http_code = 0;
				curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

				// 处理重定向 (3xx)
				if (http_code >= 300 && http_code < 400) {
					char* new_url = nullptr;
					curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &new_url);
					if (new_url && strlen(new_url) > 0) {
						std::cout << "Redirect " << (redirect_count + 1) << ": "
							<< current_url << "  ->  " << new_url << std::endl;
						current_url = new_url;
						m_last_redirect_url = current_url;
						continue; // 用新 URL 重试
					}
					// 3xx 但无 Location 头
					last_error_code = http_code;
					m_last_error_msg = "HTTP redirect without Location header";
					appendRedirectInfo();
					fs::remove(m_file_path);
					curl_easy_cleanup(curl);
					return false;
				}

				// HTTP 错误 (>= 400)
				if (http_code >= 400) {
					last_error_code = static_cast<int>(http_code);
					m_last_error_msg = "HTTP error: " + std::to_string(http_code);
					appendRedirectInfo();
					fs::remove(m_file_path);
					curl_easy_cleanup(curl);
					return false;
				}

				// 成功 (2xx)
				if (!m_last_redirect_url.empty())
					std::cout << "Download succeeded (final URL: " << m_last_redirect_url << ")" << std::endl;
				else
					std::cout << "Download succeeded" << std::endl;
				success = true;
				break;
			}

			if (!success) {
				last_error_code = -3;
				if (!m_last_redirect_url.empty())
					m_last_error_msg = "Exceeded max redirects (last: " + m_last_redirect_url + ")";
				else
					m_last_error_msg = "Exceeded max redirects";
				fs::remove(m_file_path);
				curl_easy_cleanup(curl);
				return false;
			}

			curl_easy_cleanup(curl);
			last_error_code = 0;
			m_last_error_msg = "Download succeeded";
			return true;
		}

		// 将最后的重定向 URL 追加到错误信息末尾
		void appendRedirectInfo() {
			if (!m_last_redirect_url.empty())
				m_last_error_msg += " (last redirect: " + m_last_redirect_url + ")";
		}

		// ---- sid参数 ----

		// 生成UUID（基于Windows UuidCreate）
		static std::string generateUUID() {
			UUID uuid;
			UuidCreate(&uuid);
			RPC_CSTR rpc_str;
			if (UuidToStringA(&uuid, &rpc_str) == RPC_S_OK) {
				std::string result(reinterpret_cast<char*>(rpc_str));
				RpcStringFreeA(&rpc_str);
				return result;
			}
			return "";
		}

		// 为URL追加sid={uuid}参数
		static std::string appendSidParam(const std::string& url) {
			std::string sid = generateUUID();
			if (url.find('?') != std::string::npos)
				return url + "&sid=" + sid;
			else
				return url + "?sid=" + sid;
		}
	};
}