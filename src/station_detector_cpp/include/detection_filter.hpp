#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <vector>
#include <unordered_map>

/**
 * DetectionFilter
 *
 * 等价于 Python 中的 DetectionValidator：
 * - 通过历史中心点过滤不稳定检测
 * - 拒绝大跳变
 * - 保持检测连续性
 */
class DetectionFilter
{
public:
    /**
     * 构造函数
     *
     * @param max_jump_distance        最大允许跳跃距离（像素）
     * @param min_consistent_detections 最小连续有效检测次数
     * @param history_size             历史记录长度
     */
    DetectionFilter(float max_jump_distance        = 100.0f,
                    int   min_consistent_detections = 2,
                    int   history_size              = 5);

    /**
     * 验证单个检测是否有效
     *
     * 对应 Python:
     *   validate(self, detection, image_size)
     *
     * @param center_x     检测中心 x
     * @param center_y     检测中心 y
     * @param confidence   置信度
     * @param image_width  图像宽
     * @param image_height 图像高
     * @return             是否通过过滤
     */
    bool validate(float center_x,
                  float center_y,
                  float confidence,
                  int   image_width,
                  int   image_height);

    /// 重置内部状态（对应 Python reset）
    void reset();

    /// 获取历史平均位置（少于 2 个历史点时返回最近一次有效中心或 (0,0)）
    cv::Point2f getAveragePosition() const;

    /// 当前是否已经有有效历史（last_valid_center 是否存在）
    bool hasValidCenter() const { return has_last_valid_center_; }

    /// 获取最近一次有效中心（若不存在则返回 0,0）
    cv::Point2f getLastValidCenter() const;

    /// 获取当前连续有效检测次数
    int getConsistentCount() const { return consistent_count_; }

private:
    float      max_jump_distance_;
    int        min_consistent_detections_;
    int        history_size_;
    std::deque<cv::Point2f> history_;
    int        consistent_count_;
    cv::Point2f last_valid_center_;
    bool        has_last_valid_center_;
};

/**
 * MultiDetectionTracker
 *
 * 等价于 Python detection_validator.py 的 MultiDetectionTracker：
 * - 维护多个 track（center/age/hits）
 * - 将本帧 detections 与已有 tracks 按距离进行关联
 * - 返回 hits 多且 age 小的最佳 track 对应的 detection 索引
 */
class MultiDetectionTracker
{
public:
    explicit MultiDetectionTracker(float max_track_distance = 150.0f)
        : max_track_distance_(max_track_distance) {}

    /**
     * @param centers  本帧所有候选中心点
     * @return         最佳 detection 索引；若没有则返回 -1
     */
    int update(const std::vector<cv::Point2f>& centers);

    void reset();

private:
    struct Track {
        cv::Point2f center{0.0f, 0.0f};
        int age{0};
        int hits{0};
    };

    float max_track_distance_;
    std::unordered_map<int, Track> tracks_;
    int next_track_id_{0};
};