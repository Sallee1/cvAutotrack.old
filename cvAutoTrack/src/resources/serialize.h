#pragma once
#include "pch.h"

namespace TianLi::Utils
{
	class serializeStream
	{
	private:
		std::ostream& fileOut;
	public:
		//将流对齐到size（2的幂次）的整数倍
		void align(int size = 4) {
			int p = 0;
			for (p = 0; size != 1; p++)
				size >>= 1;
			size <<= p;

			size_t fsLen = fileOut.tellp();
			size_t addLen = (fsLen >> p << p) + size - fsLen;
			if (addLen != size) {
				std::vector<char> emptyBinarys(addLen, 0);
				fileOut.write(emptyBinarys.data(), addLen);
			}
		}

		serializeStream(std::ostream& fileOut) :fileOut(fileOut) {}

		void operator <<(const std::string& s)
		{
			align(sizeof(DWORD));
			DWORD size = static_cast<DWORD>(s.size() + 1);
			fileOut.write((char*)&size, sizeof(DWORD));
			fileOut.write(s.c_str(), size);
		}

		template<typename Digital>
			requires std::is_integral_v<Digital> || std::is_floating_point_v<Digital>
		void operator <<(const Digital & d)
		{
			align(sizeof(Digital));
			fileOut.write((char*)&d, sizeof(d));
		}

		void operator <<(const std::vector<cv::KeyPoint>& points)
		{
			align(sizeof(DWORD));
			DWORD size = static_cast<DWORD>(points.size() * sizeof(cv::KeyPoint));
			fileOut.write((char*)&size, sizeof(DWORD));

			const cv::KeyPoint* keyPointPtr = points.data();
			fileOut.write((char*)keyPointPtr, size);
		}

		void operator <<(const cv::Mat& mat)
		{
			align(sizeof(DWORD));
			std::vector<uchar> buf;
			cv::imencode(".tiff", mat, buf, std::vector<int>{cv::IMWRITE_TIFF_COMPRESSION, cv::IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE});

			DWORD size = static_cast<DWORD>(buf.size());
			fileOut.write((char*)&size, sizeof(DWORD));

			const uchar* bytes = buf.data();
			fileOut.write((char*)bytes, size);
		}

		// New helpers for common OpenCV/STD types
		void operator <<(const cv::Rect2i& r)
		{
			*this << r.x; *this << r.y; *this << r.width; *this << r.height;
		}

		void operator <<(const cv::Point2i& p)
		{
			*this << p.x; *this << p.y;
		}

		void operator <<(const cv::Size2i& s)
		{
			*this << s.width; *this << s.height;
		}

		void operator <<(const std::vector<int>& v)
		{
			align(sizeof(DWORD));
			DWORD size = static_cast<DWORD>(v.size());
			fileOut.write((char*)&size, sizeof(DWORD));
			if (!v.empty())
			{
				fileOut.write((const char*)v.data(), sizeof(int) * v.size());
			}
		}

		void operator <<(const std::vector<std::vector<int>>& vv)
		{
			align(sizeof(DWORD));
			DWORD outer = static_cast<DWORD>(vv.size());
			fileOut.write((char*)&outer, sizeof(DWORD));
			for (const auto& v : vv)
			{
				*this << v;
			}
		}

		void operator <<(const std::vector<cv::Rect2i>& vr)
		{
			align(sizeof(DWORD));
			DWORD size = static_cast<DWORD>(vr.size());
			fileOut.write((char*)&size, sizeof(DWORD));
			for (const auto& r : vr)
			{
				*this << r;
			}
		}
	};

	class deSerializeStream
	{
	private:
		std::istream& fileIn;
	public:
		deSerializeStream(std::istream& fileIn) :fileIn(fileIn) {}

		void align(int size = 4)		//将流对齐到size（2的幂次）的整数倍
		{
			int p = 0;
			for (p = 0; size != 1; p++)
				size >>= 1;
			size <<= p;

			size_t fsLen = fileIn.tellg();
			size_t addLen = (fsLen >> p << p) + size - fsLen;
			if (addLen != size) {
				fileIn.ignore(addLen);
			}
		}

		void operator >>(std::string& s)
		{
			align(sizeof(DWORD));
			DWORD size;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&size, sizeof(DWORD));
			if (size > MAXSHORT) throw std::exception("解析到的字符串过长");

			std::vector<char> bytes(size);
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read(bytes.data(), size);

			s = std::string(bytes.data());
		}

		template<typename Digital>
			requires std::is_integral_v<Digital> || std::is_floating_point_v<Digital>
		void operator >>(Digital & d)
		{
			align(sizeof(Digital));
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&d, sizeof(d));
		}

		void operator >>(std::vector<cv::KeyPoint>& points)
		{
			align(sizeof(DWORD));
			DWORD size;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&size, sizeof(DWORD));

			std::vector<char> bytes(size);
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read(bytes.data(), size);

			cv::KeyPoint* keyPointPtr = (cv::KeyPoint*)bytes.data();
			points = std::vector<cv::KeyPoint>(keyPointPtr, keyPointPtr + size / sizeof(cv::KeyPoint));
		}

		void operator >>(cv::Mat& mat)
		{
			align(sizeof(DWORD));
			DWORD size;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&size, sizeof(DWORD));

			std::vector<char> bytes(size);
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read(bytes.data(), size);

			std::vector<uchar> inputArray(bytes.begin(), bytes.end());
			mat = cv::imdecode(inputArray, cv::IMREAD_ANYDEPTH | cv::IMREAD_UNCHANGED);
		}

		// New helpers for common OpenCV/STD types
		void operator >>(cv::Rect2i& r)
		{
			*this >> r.x; *this >> r.y; *this >> r.width; *this >> r.height;
		}

		void operator >>(cv::Point2i& p)
		{
			*this >> p.x; *this >> p.y;
		}

		void operator >>(cv::Size2i& s)
		{
			*this >> s.width; *this >> s.height;
		}

		void operator >>(std::vector<int>& v)
		{
			align(sizeof(DWORD));
			DWORD size;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&size, sizeof(DWORD));
			v.resize(size);
			if (size > 0)
			{
				if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
				fileIn.read((char*)v.data(), sizeof(int) * size);
			}
		}

		void operator >>(std::vector<std::vector<int>>& vv)
		{
			align(sizeof(DWORD));
			DWORD outer;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&outer, sizeof(DWORD));
			vv.resize(outer);
			for (DWORD i = 0; i < outer; ++i)
			{
				*this >> vv[i];
			}
		}

		void operator >>(std::vector<cv::Rect2i>& vr)
		{
			align(sizeof(DWORD));
			DWORD size;
			if (fileIn.eof()) throw std::exception("尝试读取已经为空的流");
			fileIn.read((char*)&size, sizeof(DWORD));
			vr.resize(size);
			for (DWORD i = 0; i < size; ++i)
			{
				*this >> vr[i];
			}
		}

		// expose eof check
		bool eof() const { return fileIn.eof(); }
		std::istream& stream() { return fileIn; }
	};
}