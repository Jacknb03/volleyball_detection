#include "yolo_inference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

bool dnnCudaAvailable()
{
    const auto backends = cv::dnn::getAvailableBackends();
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA &&
            (backend.second == cv::dnn::DNN_TARGET_CUDA ||
             backend.second == cv::dnn::DNN_TARGET_CUDA_FP16))
        {
            return true;
        }
    }
    return false;
}

void applyCpuBackend(cv::dnn::Net& net)
{
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void applyCudaBackend(cv::dnn::Net& net)
{
    bool use_fp16 = false;
    for (const auto& backend : cv::dnn::getAvailableBackends()) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA &&
            backend.second == cv::dnn::DNN_TARGET_CUDA_FP16)
        {
            use_fp16 = true;
            break;
        }
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(use_fp16 ? cv::dnn::DNN_TARGET_CUDA_FP16
                                     : cv::dnn::DNN_TARGET_CUDA);
}

}  // namespace

YOLODetector::YOLODetector(const std::string& model_path,
                           const std::string& model_type,
                           float conf_threshold,
                           float iou_threshold,
                           const std::string& device)
    : model_path_(model_path),
      model_type_(model_type),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      device_(device)
{
    // 对标 Python: __init__ 记录参数 + 加载模型。
    // 这里使用 OpenCV DNN 加载 ONNX 模型（model_path 必须指向 .onnx）。
    class_names_ = coco80Names();
    num_classes_ = static_cast<int>(class_names_.size());
}

std::vector<std::string> YOLODetector::coco80Names()
{
    // Standard COCO-80 class names used by YOLOv5/YOLOv8 pretrained models.
    return {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };
}

std::string YOLODetector::getClassName(int class_id) const
{
    if (class_id < 0) {
        return "class";
    }
    if (class_id < static_cast<int>(class_names_.size())) {
        return class_names_[class_id];
    }
    return "class_" + std::to_string(class_id);
}

void YOLODetector::ensureNetInitialized() const
{
    if (net_initialized_) {
        return;
    }

    if (model_path_.empty()) {
        throw std::runtime_error(
            "YOLODetector requires an ONNX model path. "
            "Set parameter yolo.model_path to a .onnx file (exported from YOLOv5/YOLOv8).");
    }

    net_ = cv::dnn::readNet(model_path_);

    // Device selection: probe OpenCV DNN at runtime (do not rely on compile-time HAVE_CUDA).
    std::string dev = device_;
    std::transform(dev.begin(), dev.end(), dev.begin(), ::tolower);

    const bool want_cuda = (dev == "cuda" || dev == "auto");
    const bool cuda_ok = dnnCudaAvailable();

    if (force_cpu_backend_ || dev == "cpu" || !want_cuda || !cuda_ok) {
        applyCpuBackend(net_);
    } else {
        applyCudaBackend(net_);
    }

    net_initialized_ = true;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) const
{
    // 对标 Python: detect(self, image) -> List[dict]
    // - 这里不做预处理/归一化等细节，由 runModel 负责
    // - 只负责调用模型并返回 Detection 列表
    std::vector<Detection> results;

    if (image.empty()) {
        return results;
    }

    runModel(image, results);

    // Python 版本里在 YOLOv5 路径中还会做一次 conf >= self.conf_threshold 过滤；
    // C++ 版本保持相同行为：如果模型推理未过滤，则这里补一层。
    if (!results.empty()) {
        results.erase(
            std::remove_if(results.begin(), results.end(),
                           [this](const Detection& d) {
                               return d.confidence < this->conf_threshold_;
                           }),
            results.end());
    }

    return results;
}

std::vector<Detection> YOLODetector::filterByClassIds(
    const std::vector<Detection>& detections,
    const std::vector<int>& allowed_class_ids) const
{
    // Python filter_volleyball 的等价实现（只是从“按名称”改成“按 ID 集合”）
    if (detections.empty() || allowed_class_ids.empty()) {
        return {};
    }

    std::vector<Detection> filtered;
    filtered.reserve(detections.size());

    for (const auto& det : detections) {
        if (std::find(allowed_class_ids.begin(),
                      allowed_class_ids.end(),
                      det.class_id) != allowed_class_ids.end())
        {
            filtered.push_back(det);
        }
    }

    return filtered;
}

bool YOLODetector::selectBestDetection(const std::vector<Detection>& detections,
                                       const std::string& method,
                                       Detection& out_detection) const
{
    // 对标 Python:
    // - method == "confidence" -> max(detections, key=confidence)
    // - method == "center"     -> 注释里提到“按中心距离”，实际实现仍然按置信度
    if (detections.empty()) {
        return false;
    }

    // 与 Python 一致：无论 "confidence" 还是 "center"，都选置信度最高的
    auto it = std::max_element(
        detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence < b.confidence;
        });

    if (it == detections.end()) {
        return false;
    }

    out_detection = *it;
    return true;
}

void YOLODetector::drawDetections(cv::Mat& image,
                                  const std::vector<Detection>& detections,
                                  const cv::Scalar& color) const
{
    // 对标 Python draw_detections:
    // - 绘制矩形框
    // - 绘制中心点
    // - 绘制 (class_name, conf) 文本
    //
    // 由于 Detection 中没有 class_name，这里仅绘制置信度。
    if (image.empty() || detections.empty()) {
        return;
    }

    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.x);
        int y1 = static_cast<int>(det.y);
        int x2 = static_cast<int>(det.x + det.width);
        int y2 = static_cast<int>(det.y + det.height);

        // 边界检查（防止超出图像范围导致 OpenCV 抛异常）
        x1 = std::max(0, std::min(x1, image.cols - 1));
        y1 = std::max(0, std::min(y1, image.rows - 1));
        x2 = std::max(0, std::min(x2, image.cols - 1));
        y2 = std::max(0, std::min(y2, image.rows - 1));

        // 绘制边界框
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // 绘制中心点
        int cx = static_cast<int>(det.x + det.width  * 0.5f);
        int cy = static_cast<int>(det.y + det.height * 0.5f);
        cx = std::max(0, std::min(cx, image.cols - 1));
        cy = std::max(0, std::min(cy, image.rows - 1));

        cv::circle(image, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);

        // 绘制置信度文本
        char buf[64];
        std::snprintf(buf, sizeof(buf), "conf: %.2f", static_cast<double>(det.confidence));
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(buf,
                                             cv::FONT_HERSHEY_SIMPLEX,
                                             0.5, 1, &baseline);

        int text_x = x1;
        int text_y = std::max(0, y1 - 5);

        // 防止文字超出顶部
        if (text_y - text_size.height < 0) {
            text_y = y1 + text_size.height + 5;
        }

        cv::putText(image, buf,
                    cv::Point(text_x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1);
    }
}

void YOLODetector::updateThresholds(float conf_threshold, float iou_threshold)
{
    // 对标 Python update_thresholds()
    conf_threshold_ = conf_threshold;
    iou_threshold_  = iou_threshold;

    // 在 Python YOLOv5 分支中, self.model.conf / self.model.iou 会被一起更新，
    // 这里不直接操作模型实例，由具体 runModel 实现去使用这两个成员变量。
}

static inline float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

void YOLODetector::runModel(const cv::Mat& image,
                            std::vector<Detection>& detections) const
{
    detections.clear();
    ensureNetInitialized();

    // Typical YOLO ONNX export uses 640x640; to avoid redesign, keep fixed 640.
    // If your export uses different size, adjust here to match the ONNX model.
    const int inp_w = 640;
    const int inp_h = 640;

    const float x_factor = static_cast<float>(image.cols) / static_cast<float>(inp_w);
    const float y_factor = static_cast<float>(image.rows) / static_cast<float>(inp_h);

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(inp_w, inp_h),
                           cv::Scalar(), true, false); // swapRB=true, crop=false

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    } catch (const cv::Exception&) {
        if (!force_cpu_backend_) {
            force_cpu_backend_ = true;
            net_initialized_ = false;
            ensureNetInitialized();
            net_.setInput(blob);
            net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        } else {
            throw;
        }
    }
    if (outputs.empty()) {
        return;
    }

    // Merge outputs if multiple heads exist (some exports return one tensor, some several).
    // We parse both [1,C,N] and [1,N,C], where C is expected to be (4 + num_classes).
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    auto updateClassMetadata = [this](int cls_count) {
        if (cls_count <= 0) {
            return;
        }
        num_classes_ = cls_count;
        if (cls_count == 1) {
            class_names_ = {"volleyball"};
            return;
        }
        const auto coco = coco80Names();
        class_names_.clear();
        class_names_.reserve(cls_count);
        for (int i = 0; i < cls_count; ++i) {
            if (i < static_cast<int>(coco.size())) {
                class_names_.push_back(coco[i]);
            } else {
                class_names_.push_back("class_" + std::to_string(i));
            }
        }
    };

    auto pushCandidate = [&](float cx, float cy, float w, float h, int best_cls, float conf) {
        if (conf < conf_threshold_) {
            return;
        }
        float left = (cx - 0.5f * w) * x_factor;
        float top  = (cy - 0.5f * h) * y_factor;
        float width = w * x_factor;
        float height = h * y_factor;

        left = clampf(left, 0.0f, static_cast<float>(image.cols - 1));
        top  = clampf(top,  0.0f, static_cast<float>(image.rows - 1));
        width  = clampf(width,  0.0f, static_cast<float>(image.cols) - left);
        height = clampf(height, 0.0f, static_cast<float>(image.rows) - top);
        if (width <= 1.0f || height <= 1.0f) {
            return;
        }

        boxes.emplace_back(cv::Rect(static_cast<int>(left),
                                    static_cast<int>(top),
                                    static_cast<int>(width),
                                    static_cast<int>(height)));
        scores.push_back(conf);
        class_ids.push_back(best_cls);
    };

    auto parseNcRows = [&](const cv::Mat& row_major) {
        // row_major: [N, C], C should be 4 + num_classes
        if (row_major.empty() || row_major.rows <= 0 || row_major.cols < 5) {
            return;
        }
        const int C = row_major.cols;
        const int cls_count = C - 4;
        if (cls_count <= 0) {
            return;
        }
        updateClassMetadata(cls_count);

        for (int i = 0; i < row_major.rows; ++i) {
            const float* data = row_major.ptr<float>(i);
            if (!data) {
                continue;
            }
            const float cx = data[0];
            const float cy = data[1];
            const float w  = data[2];
            const float h  = data[3];

            // Custom YOLOv8 [1, 5, 8400]: class score at index 4.
            int best_cls = 0;
            float best_prob = data[4];
            for (int c = 1; c < cls_count; ++c) {
                const float p = data[4 + c];
                if (p > best_prob) {
                    best_prob = p;
                    best_cls = c;
                }
            }
            const float conf = best_prob;
            pushCandidate(cx, cy, w, h, best_cls, conf);
        }
    };

    auto parse_one = [&](const cv::Mat& out) {
        if (out.empty()) {
            return;
        }
        if (out.dims == 3) {
            // Expected: [1, C, N] or [1, N, C]
            if (out.size[0] != 1) {
                return;
            }
            const int d1 = out.size[1];
            const int d2 = out.size[2];

            // Case A: [1, C, N], and C should be >=5 (4 + num_classes)
            if (d1 >= 5 && d2 > 0) {
                const int C = d1;
                const int N = d2;
                const int cls_count = C - 4;
                if (cls_count <= 0) {
                    return;
                }
                updateClassMetadata(cls_count);

                for (int i = 0; i < N; ++i) {
                    const float cx = out.at<float>(0, 0, i);
                    const float cy = out.at<float>(0, 1, i);
                    const float w  = out.at<float>(0, 2, i);
                    const float h  = out.at<float>(0, 3, i);

                    int best_cls = 0;
                    float best_prob = out.at<float>(0, 4, i); // index 4 alignment
                    for (int c = 1; c < cls_count; ++c) {
                        const float p = out.at<float>(0, 4 + c, i);
                        if (p > best_prob) {
                            best_prob = p;
                            best_cls = c;
                        }
                    }
                    pushCandidate(cx, cy, w, h, best_cls, best_prob);
                }
                return;
            }

            // Case B: [1, N, C]
            if (d2 >= 5 && d1 > 0) {
                cv::Mat row_major = out.reshape(1, d1); // [N, C]
                parseNcRows(row_major);
            }
            return;
        }

        if (out.dims == 2) {
            // Could be [N, C] or [C, N]
            cv::Mat row_major = out;
            if (row_major.cols < 5 && row_major.rows >= 5) {
                cv::Mat t;
                cv::transpose(row_major, t);
                row_major = t;
            }
            parseNcRows(row_major);
            return;
        }
    };

    for (auto& out : outputs) {
        // Ensure float type
        cv::Mat out_f;
        if (out.type() != CV_32F) {
            out.convertTo(out_f, CV_32F);
        } else {
            out_f = out;
        }
        parse_one(out_f);
    }

    if (boxes.empty()) {
        return;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, iou_threshold_, indices);

    detections.reserve(indices.size());
    for (int idx : indices) {
        const auto& r = boxes[idx];
        Detection d;
        d.x = static_cast<float>(r.x);
        d.y = static_cast<float>(r.y);
        d.width = static_cast<float>(r.width);
        d.height = static_cast<float>(r.height);
        d.confidence = scores[idx];
        d.class_id = class_ids[idx];
        detections.push_back(d);
    }
}
