#include "detection_filter.hpp"

#include <cmath>
#include <limits>
#include <algorithm>

DetectionFilter::DetectionFilter(float max_jump_distance,
                                 int   min_consistent_detections,
                                 int   history_size)
    : max_jump_distance_(max_jump_distance),
      min_consistent_detections_(min_consistent_detections),
      history_size_(history_size),
      consistent_count_(0),
      last_valid_center_(0.0f, 0.0f),
      has_last_valid_center_(false)
{
    if (history_size_ < 1) {
        history_size_ = 1;
    }
}

bool DetectionFilter::validate(float center_x,
                               float center_y,
                               float confidence,
                               int   image_width,
                               int   image_height)
{
    // 对应 Python validate():
    // 1) 检查 detection 是否存在
    // 2) 检查中心在图像范围内
    // 3) 检查置信度 >= 0.3
    // 4) 检查与历史位置的跳变
    // 5) 更新历史与连续计数
    // 6) 检查连续检测次数

    // 1. 基本检查（这里假定调用方只在有检测时调用，
    //    因此不再处理 detection == None 的情况）

    // 2. 中心点是否在图像范围内
    if (center_x < 0.0f || center_x >= static_cast<float>(image_width) ||
        center_y < 0.0f || center_y >= static_cast<float>(image_height))
    {
        return false;
    }

    // 3. 置信度下限（与 Python 固定 0.3 一致）
    if (confidence < 0.3f) {
        return false;
    }

    cv::Point2f center(center_x, center_y);

    // 4. 检查与上一次有效中心的跳变距离
    if (has_last_valid_center_) {
        float dx = center.x - last_valid_center_.x;
        float dy = center.y - last_valid_center_.y;
        float distance = std::sqrt(dx * dx + dy * dy);

        if (distance > max_jump_distance_) {
            // 跳变过大，认为是误检，重置连续计数
            consistent_count_ = 0;
            return false;
        }
    }

    // 5. 更新历史记录
    if (static_cast<int>(history_.size()) >= history_size_) {
        history_.pop_front();
    }
    history_.push_back(center);

    consistent_count_ += 1;
    last_valid_center_    = center;
    has_last_valid_center_ = true;

    // 6. 连续检测次数不足时仍返回 false（需“热身”若干帧）
    if (consistent_count_ < min_consistent_detections_) {
        return false;
    }

    return true;
}

void DetectionFilter::reset()
{
    history_.clear();
    consistent_count_      = 0;
    last_valid_center_     = cv::Point2f(0.0f, 0.0f);
    has_last_valid_center_ = false;
}

cv::Point2f DetectionFilter::getAveragePosition() const
{
    // 对应 Python get_average_position():
    // - 若历史长度 < 2，则返回 last_valid_center
    // - 否则返回历史均值
    if (history_.size() < 2) {
        if (has_last_valid_center_) {
            return last_valid_center_;
        }
        return cv::Point2f(0.0f, 0.0f);
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (const auto& p : history_) {
        sum_x += p.x;
        sum_y += p.y;
    }
    float n = static_cast<float>(history_.size());
    return cv::Point2f(sum_x / n, sum_y / n);
}

cv::Point2f DetectionFilter::getLastValidCenter() const
{
    if (has_last_valid_center_) {
        return last_valid_center_;
    }
    return cv::Point2f(0.0f, 0.0f);
}

int MultiDetectionTracker::update(const std::vector<cv::Point2f>& centers)
{
    // 对齐 Python MultiDetectionTracker.update(detections):
    // - 没有检测：age++，age>5 删除，返回 None
    // - 关联：对每个 track 找到距离最近且 < max_track_distance 的 center
    // - 未匹配的 center 新建 track
    // - 删除 age>5 track
    // - 选 hits 最多且 age 最小的 track
    // - 返回与该 track center 距离 < max_track_distance 的某个 detection（这里返回最近的索引）

    if (centers.empty()) {
        for (auto it = tracks_.begin(); it != tracks_.end(); ) {
            it->second.age += 1;
            if (it->second.age > 5) {
                it = tracks_.erase(it);
            } else {
                ++it;
            }
        }
        return -1;
    }

    std::vector<bool> matched(centers.size(), false);

    // match existing tracks
    for (auto& kv : tracks_) {
        auto& track = kv.second;
        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < centers.size(); ++i) {
            if (matched[i]) continue;
            float dx = centers[i].x - track.center.x;
            float dy = centers[i].y - track.center.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < max_track_distance_ && dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0) {
            track.center = centers[best_idx];
            track.age = 0;
            track.hits += 1;
            matched[best_idx] = true;
        }
    }

    // create tracks for unmatched detections
    for (size_t i = 0; i < centers.size(); ++i) {
        if (!matched[i]) {
            Track t;
            t.center = centers[i];
            t.age = 0;
            t.hits = 1;
            tracks_[next_track_id_++] = t;
        }
    }

    // delete expired tracks
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second.age > 5) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }

    if (tracks_.empty()) {
        return -1;
    }

    // best track: (hits max, age min) == Python key (hits, -age)
    int best_track_id = -1;
    int best_hits = -1;
    int best_age = std::numeric_limits<int>::max();

    for (const auto& kv : tracks_) {
        const auto& tr = kv.second;
        if (tr.hits > best_hits || (tr.hits == best_hits && tr.age < best_age)) {
            best_hits = tr.hits;
            best_age = tr.age;
            best_track_id = kv.first;
        }
    }

    if (best_track_id < 0) {
        return -1;
    }

    const auto& best_track = tracks_.at(best_track_id);

    // return the closest center to best_track.center within max_track_distance
    int best_det_idx = -1;
    float best_det_dist = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < centers.size(); ++i) {
        float dx = centers[i].x - best_track.center.x;
        float dy = centers[i].y - best_track.center.y;
        float dist = std::sqrt(dx*dx + dy*dy);
        if (dist < max_track_distance_ && dist < best_det_dist) {
            best_det_dist = dist;
            best_det_idx = static_cast<int>(i);
        }
    }

    return best_det_idx;
}

void MultiDetectionTracker::reset()
{
    tracks_.clear();
    next_track_id_ = 0;
}