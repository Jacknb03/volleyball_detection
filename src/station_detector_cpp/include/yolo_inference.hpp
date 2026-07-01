#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <stdexcept>

/**
 * 检测结果结构体
 * 对应 Python 里的:
 * {
 *   'bbox': [x1, y1, x2, y2],
 *   'confidence': float,
 *   'class_id': int,
 * }
 *
 * 这里用 (x, y, width, height) 表示边界框。
 */
struct Detection {
    float x;          // 左上角 x
    float y;          // 左上角 y
    float width;      // 宽度
    float height;     // 高度
    float confidence; // 置信度
    int   class_id;   // 类别 ID
};

/**
 * YOLO 检测器 C++ 封装
 *
 * 目标：
 * - public 接口与 Python 版逻辑对应（detect / 选择最佳框 / 阈值更新 / 绘制等）
 * - 不在此处重新设计算法；具体模型推理由子类或外部实现 runModel()
 *
 * 使用方式（示例）：
 * - 派生类重写 runModel()，在其中调用 YOLOv5/YOLOv8 C++/ONNX/libtorch 等推理，
 *   填充 Detection 向量。
 */
class YOLODetector
{
public:
    /**
     * 构造函数
     *
     * @param model_path  模型路径；为空时使用默认预训练模型（由具体实现决定）
     * @param model_type  "yolov5" 或 "yolov8"
     * @param conf_threshold  置信度阈值
     * @param iou_threshold   NMS IoU 阈值
     * @param device          "cpu" / "cuda" / "auto"（由具体实现解释）
     */
    YOLODetector(const std::string& model_path = "",
                 const std::string& model_type = "yolov8",
                 float conf_threshold = 0.5f,
                 float iou_threshold  = 0.45f,
                 const std::string& device = "auto",
                 int input_size = 640);

    virtual ~YOLODetector() = default;

    /**
     * 在一帧图像上运行检测
     *
     * @param image  BGR 格式 cv::Mat
     * @return       检测结果列表
     */
    std::vector<Detection> detect(const cv::Mat& image) const;

    /**
     * 按类别 ID 过滤检测结果
     * （Python 的 filter_volleyball 是按类别名，这里等价为“只保留给定 ID 集合”）
     */
    std::vector<Detection> filterByClassIds(
        const std::vector<Detection>& detections,
        const std::vector<int>& allowed_class_ids) const;

    /**
     * 从多个检测中选择最佳一个
     *
     * @param detections   候选列表
     * @param method       "confidence" 或 "center"
     *                     - "confidence": 选择置信度最大的（与 Python 相同）
     *                     - "center": 与 Python 一样，目前也退化为按置信度选择
     * @param out_detection 输出：最佳检测；如果返回 false 则内容未定义
     * @return              是否找到检测（detections 非空）
     */
    bool selectBestDetection(const std::vector<Detection>& detections,
                             const std::string& method,
                             Detection& out_detection) const;

    /**
     * 在图像上绘制检测结果（与 Python draw_detections 对齐）
     *
     * @param image       输入/输出图像（BGR）
     * @param detections  要绘制的检测结果
     * @param color       边框颜色（默认绿色）
     */
    void drawDetections(cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const cv::Scalar& color = cv::Scalar(0, 255, 0)) const;

    /**
     * 更新检测阈值（与 Python update_thresholds 对齐）
     */
    void updateThresholds(float conf_threshold, float iou_threshold);

    float getConfThreshold() const { return conf_threshold_; }
    float getIouThreshold() const  { return iou_threshold_; }
    int getInputSize() const { return input_size_; }
    const std::string& getModelType() const { return model_type_; }
    const std::string& getDevice() const     { return device_; }
    int getNumClasses() const { return num_classes_; }
    std::string getClassName(int class_id) const;

protected:
    /**
     * 运行实际 YOLO 模型推理的接口
     *
     * 说明：
     * - 这里对应 Python 中根据 model_type 调用 ultralytics / torch.hub 的那部分逻辑。
     * - 为了“不过度重新设计算法”，这里只抽象出“给一张图，返回 N 个 Detection”这一行为。
     * - 你可以在派生类中用 ONNXRuntime / libtorch / TensorRT 等实现该函数。
     *
     * @param image      输入 BGR 图像
     * @param detections 输出检测结果（需填充 Detection 数组）
     */
    virtual void runModel(const cv::Mat& image,
                          std::vector<Detection>& detections) const;

protected:
    std::string model_path_;
    std::string model_type_;
    float        conf_threshold_;
    float        iou_threshold_;
    std::string device_;
    int          input_size_;

    // OpenCV DNN backend (ONNX inference). This is the concrete inference path
    // used to match the Python YOLODetector.detect() behavior at runtime.
    mutable bool net_initialized_{false};
    mutable bool force_cpu_backend_{false};
    mutable cv::dnn::Net net_;
    mutable int num_classes_{80}; // default COCO-80
    mutable std::vector<std::string> class_names_;

    void ensureNetInitialized() const;
public:
    static std::vector<std::string> coco80Names();
};
