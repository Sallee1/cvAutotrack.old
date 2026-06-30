#pragma once
#include "cfiledownloader.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <filesystem>

namespace tianli {
    class FileDownloaderAsync {
    public:
        explicit FileDownloaderAsync(int threadNum = 32) : m_thread_num(threadNum) {
            for (int i = 0; i < m_thread_num; ++i) {
                m_workers.emplace_back(&FileDownloaderAsync::workerThread, this);
            }
        }

        ~FileDownloaderAsync() {
            m_stop = true;
            m_queue_cond.notify_all();
            for (auto& worker : m_workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        size_t addTask(const std::string& filePath,
            const std::string& url,
            const std::string& md5 = "") {
            auto task = std::make_shared<DownloadTask>();
            task->id = m_next_task_id++;
            task->filePath = filePath;
            task->url = url;
            task->md5 = md5;

            {
                std::lock_guard<std::mutex> lock(m_tasks_mutex);
                m_tasks[task->id] = task;
            }

            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_task_queue.push(task);
            }
            m_queue_cond.notify_one();
            return task->id;
        }

        void wait() {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_wait_cond.wait(lock, [this]() {
                return m_task_queue.empty() && m_active_tasks == 0;
            });
        }

        size_t getTaskCount() const {
            std::lock_guard<std::mutex> lock(m_tasks_mutex);
            return m_tasks.size();
        }

        size_t getCompletedCount() const {
            std::lock_guard<std::mutex> lock(m_tasks_mutex);
            size_t count = 0;
            for (const auto& [id, task] : m_tasks) {
                if (task->completed) count++;
            }
            return count;
        }

        std::unordered_map<size_t, std::pair<int, std::string>> getFailed() const {
            std::unordered_map<size_t, std::pair<int, std::string>> failedTasks;
            std::lock_guard<std::mutex> lock(m_tasks_mutex);
            for (const auto& [id, task] : m_tasks) {
                if (task->completed && !task->success) {
                    failedTasks[id] = { task->errorCode, task->errorMsg };
                }
            }
            return failedTasks;
        }

        std::string getTaskDetail(size_t taskId) const {
            std::lock_guard<std::mutex> lock(m_tasks_mutex);
            auto it = m_tasks.find(taskId);
            if (it == m_tasks.end()) {
                return "Task not found";
            }
            const auto& task = it->second;
            return "ID: " + std::to_string(task->id) +
                "\nFile: " + task->filePath +
                "\nURL: " + task->url +
                "\nStatus: " + (task->completed ? (task->success ? "Success" : "Failed") : "In progress") +
                (task->completed && !task->success ? "\nError: " + std::to_string(task->errorCode) + " - " + task->errorMsg : "");
        }

        void retry() {
            std::vector<std::shared_ptr<DownloadTask>> failedTasks;
            {
                std::lock_guard<std::mutex> lock(m_tasks_mutex);
                for (auto& [id, task] : m_tasks) {
                    if (task->completed && !task->success) {
                        task->completed = false;
                        task->success = false;
                        failedTasks.push_back(task);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                for (auto& task : failedTasks) {
                    m_task_queue.push(task);
                }
            }
            m_queue_cond.notify_all();
        }

    private:
        struct DownloadTask {
            size_t id{};
            std::string filePath;
            std::string url;
            std::string md5;
            std::atomic<bool> completed{ false };
            std::atomic<bool> success{ false };
            int errorCode{ 0 };
            std::string errorMsg;
        };

        // 线程池工作函数
        void workerThread() {
            while (!m_stop) {
                std::shared_ptr<DownloadTask> task;
                {                
                    std::unique_lock<std::mutex> lock(m_queue_mutex);
                    m_queue_cond.wait(lock, [this]() { return m_stop || !m_task_queue.empty(); });
                    if (m_stop) return;
                    if (m_task_queue.empty()) continue;
                    task = m_task_queue.front();
                    m_task_queue.pop();
                }

                m_active_tasks++;

                try {
                    // 下载到临时文件，完成后原子重命名，避免写坏目标文件
                    std::string tmp_path = task->filePath + ".tmp";

                    FileDownloader downloader(tmp_path, task->url, task->md5);
                    task->success = downloader.download();

                    if (task->success) {
                        // 下载成功：.tmp → 目标文件（原子 rename 覆盖）
                        std::error_code ec;
                        std::filesystem::rename(tmp_path, task->filePath, ec);
                        if (ec) {
                            task->success = false;
                            task->errorCode = -3;
                            task->errorMsg = "重命名临时文件失败: " + ec.message();
                            std::filesystem::remove(tmp_path, ec);
                        }
                    } else {
                        task->errorCode = downloader.getLastErrorCode();
                        task->errorMsg = downloader.getLastErrorMsg();
                        // 下载失败：清理 .tmp
                        std::error_code ec;
                        std::filesystem::remove(tmp_path, ec);
                    }
                } catch (...) {
                    task->success = false;
                    task->errorCode = -1;
                    task->errorMsg = "Unknown error occurred during download";
                }
                task->completed = true;

                // 原子操作不需要锁
                m_active_tasks--;
                if (m_task_queue.empty() && m_active_tasks == 0) {
                    m_wait_cond.notify_all();
                }
            }
        }

        // 线程数
        const int m_thread_num;

        // 线程池
        std::vector<std::thread> m_workers;

        // 任务队列
        std::queue<std::shared_ptr<DownloadTask>> m_task_queue;
        mutable std::mutex m_queue_mutex;
        std::condition_variable m_queue_cond;

        // 任务存储
        std::unordered_map<size_t, std::shared_ptr<DownloadTask>> m_tasks;
        mutable std::mutex m_tasks_mutex;

        // 停止标志
        std::atomic<bool> m_stop{ false };

        // 任务ID计数器
        std::atomic<size_t> m_next_task_id{ 0 };

        // 活动任务计数器
        std::atomic<int> m_active_tasks{ 0 };

        // 等待条件变量
        std::condition_variable m_wait_cond;
    };
}
