// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// MindVision Camera SDK
#include <CameraApi.h>

// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// C++ system
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <stdexcept>
#include <cstdlib>
#include <ratio>  // 添加ratio头文件用于std::milli

namespace mindvision_camera
{
class MVCameraNode : public rclcpp::Node
{
public:
  explicit MVCameraNode(const rclcpp::NodeOptions & options) : Node("mv_camera", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting MVCameraNode!");

    try {
      // 初始化SDK
      CameraSdkInit(1);
      
      // 枚举设备
      int i_camera_counts = 1;
      int i_status = CameraEnumerateDevice(&t_camera_enum_list_, &i_camera_counts);
      RCLCPP_INFO(this->get_logger(), "Enumerate state = %d", i_status);
      RCLCPP_INFO(this->get_logger(), "Found camera count = %d", i_camera_counts);

      // 检查设备
      if (i_camera_counts == 0) {
        throw std::runtime_error("No camera found!");
      }

      // 相机初始化
      i_status = CameraInit(&t_camera_enum_list_, -1, -1, &h_camera_);
      RCLCPP_INFO(this->get_logger(), "Init state = %d", i_status);
      if (i_status != CAMERA_STATUS_SUCCESS) {
        throw std::runtime_error("Camera initialization failed with status: " + std::to_string(i_status));
      }

      // 获得相机的特性描述
      CameraGetCapability(h_camera_, &t_capability_);

      // 设置相机参数
      CameraSetAeState(h_camera_, false);
      CameraSetFrameSpeed(h_camera_, FRAME_SPEED_HIGH);  // 使用FRAME_SPEED_HIGH替代FRAME_SPEED_SUPER
      CameraSetTriggerMode(h_camera_, CONTINUATION);
      
      // 设置输出格式
      int fmt_status = CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_RGB8);
      if (fmt_status != CAMERA_STATUS_SUCCESS) {
          throw std::runtime_error("Failed to set output format: " + std::to_string(fmt_status));
      }

      // 设置分辨率
      tSdkImageResolution resolution = {0};
      resolution.iIndex = 0; // 使用自定义分辨率
      resolution.iWidthFOV = 1280;
      resolution.iHeightFOV = 720;
      resolution.iWidth = 1280;
      resolution.iHeight = 720;
      resolution.uBinAverageMode = 0;
      resolution.uBinSumMode = 0;
      resolution.uResampleMask = 0;
      resolution.uSkipMode = 0;

      int res_status = CameraSetImageResolution(h_camera_, &resolution);
      if (res_status != CAMERA_STATUS_SUCCESS) {
          throw std::runtime_error("Failed to set resolution: " + std::to_string(res_status));
      }

      // 声明ROS参数
      declareParameters();

      // 启动相机
      i_status = CameraPlay(h_camera_);
      if (i_status != CAMERA_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to start camera with status: " + std::to_string(i_status));
      }

      // 创建相机发布器
      bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
      auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
      camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

      // 加载相机信息
      camera_name_ = this->declare_parameter("camera_name", "mv_camera");
      camera_info_manager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
      auto camera_info_url = this->declare_parameter(
        "camera_info_url", "package://mindvision_camera/config/camera_info.yaml");
      if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
        camera_info_msg_ = camera_info_manager_->getCameraInfo();
      } else {
        RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
      }

      // 添加参数回调
      params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&MVCameraNode::parametersCallback, this, std::placeholders::_1));

      // 获取帧率参数
      frame_rate_ = this->get_parameter("frame_rate").as_double();
      if (frame_rate_ <= 0) {
        throw std::invalid_argument("Frame rate must be positive");
      }
      RCLCPP_INFO(this->get_logger(), "Frame rate set to: %.1f Hz", frame_rate_);
      
      // 启动捕获线程
      running_ = true;
      capture_thread_ = std::thread(&MVCameraNode::captureThread, this);
      
    } catch (const std::exception& e) {
      RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());
      // 确保资源清理
      cleanupResources();
      throw;
    }
  }

  ~MVCameraNode() override
  {
    RCLCPP_INFO(this->get_logger(), "Shutting down MVCameraNode...");
    running_ = false;
    
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    
    cleanupResources();
    RCLCPP_INFO(this->get_logger(), "Camera node destroyed!");
  }

private:
  void captureThread()
  {
    RCLCPP_INFO(this->get_logger(), "Capture thread started");
    
    using namespace std::chrono;
    nanoseconds frame_interval(0);
    if (frame_rate_ > 0) {
      frame_interval = duration_cast<nanoseconds>(duration<double>(1.0 / frame_rate_));
    } else {
      RCLCPP_ERROR(this->get_logger(), "Invalid frame rate: %.1f", frame_rate_);
      return;
    }
    
    auto next_frame_time = steady_clock::now();
    auto last_stat_time = steady_clock::now();
    int frame_count = 0;
    constexpr int STAT_INTERVAL_SEC = 5;
    
    while (rclcpp::ok() && running_) {
      auto frame_start_time = steady_clock::now();
      
      tSdkFrameHead frame_info;
      uint8_t* buffer = nullptr;
      bool frame_acquired = false;
      
      // 创建新的图像消息对象
      auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
      image_msg->header.frame_id = "camera_optical_frame";
      image_msg->encoding = "rgb8";
      
      // 获取图像缓冲区
      {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        
        int status = CameraGetImageBuffer(h_camera_, &frame_info, &buffer, 100);
        if (status == CAMERA_STATUS_SUCCESS) {
          frame_acquired = true;
          
          // 预分配图像数据缓冲区
          const size_t required_size = frame_info.iWidth * frame_info.iHeight * 3;
          image_msg->data.resize(required_size);
          
          // 处理图像
          CameraImageProcess(h_camera_, buffer, image_msg->data.data(), &frame_info);
          
          // 释放缓冲区
          CameraReleaseImageBuffer(h_camera_, buffer);
          buffer = nullptr;
          
          fail_count_ = 0;
        } else {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000, 
                              "Failed to get image buffer, status = %d", status);
          fail_count_++;
          
          if (fail_count_ > 10) {
            RCLCPP_FATAL(this->get_logger(), "Failed to get image buffer repeatedly, exiting!");
            rclcpp::shutdown();
            break;
          }
        }
      }
      
      // 如果没有获取到帧，直接跳到下一帧
      if (!frame_acquired) {
        next_frame_time += frame_interval;
        std::this_thread::sleep_until(next_frame_time);
        continue;
      }
      
      // 设置时间戳
      const auto now = this->now();
      image_msg->header.stamp = now;
      camera_info_msg_.header.stamp = now;
      
      // 设置图像尺寸
      image_msg->height = frame_info.iHeight;
      image_msg->width = frame_info.iWidth;
      image_msg->step = frame_info.iWidth * 3;
      camera_info_msg_.width = image_msg->width;
      camera_info_msg_.height = image_msg->height;
      
      // 验证图像数据大小
      size_t expected_size = static_cast<size_t>(image_msg->step) * image_msg->height;
      if (image_msg->data.size() < expected_size) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Image buffer too small! Required: %zu, Available: %zu", 
                    expected_size, image_msg->data.size());
        image_msg->data.resize(expected_size);
      }
      
      // 发布图像
      if (camera_pub_.getNumSubscribers() > 0) {
        try {
          // 使用move转移所有权
          camera_pub_.publish(std::move(*image_msg), camera_info_msg_);
          frame_count++;
        } catch (const std::exception& e) {
          RCLCPP_ERROR(this->get_logger(), "Publish exception: %s", e.what());
        }
      }
      
      // 性能统计
      auto current_time = steady_clock::now();
      if (duration_cast<seconds>(current_time - last_stat_time).count() >= STAT_INTERVAL_SEC) {
        double actual_fps = static_cast<double>(frame_count) / STAT_INTERVAL_SEC;
        RCLCPP_INFO(this->get_logger(), "Actual frame rate: %.1f FPS", actual_fps);
        frame_count = 0;
        last_stat_time = current_time;
      }
      
      // 帧率控制
      const auto process_time = steady_clock::now() - frame_start_time;
      if (process_time < frame_interval) {
        next_frame_time += frame_interval;
        std::this_thread::sleep_until(next_frame_time);
      } else {
        double process_time_ms = duration<double, std::milli>(process_time).count();
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000, 
                            "Frame processing took too long: %.2f ms", 
                            process_time_ms);
        next_frame_time = steady_clock::now();
      }
    }
    
    RCLCPP_INFO(this->get_logger(), "Capture thread exiting");
  }

  void cleanupResources()
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    
    if (h_camera_ != -1) {
      RCLCPP_INFO(this->get_logger(), "Stopping camera...");
      CameraStop(h_camera_);
      
      RCLCPP_INFO(this->get_logger(), "Uninitializing camera...");
      CameraUnInit(h_camera_);
      h_camera_ = -1;
    }
  }

  void declareParameters()
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;

    // 帧率参数
    param_desc.description = "Frame rate in Hz";
    param_desc.floating_point_range.resize(1);
    param_desc.floating_point_range[0].from_value = 1.0;
    param_desc.floating_point_range[0].to_value = 60.0;
    param_desc.floating_point_range[0].step = 1.0;
    frame_rate_ = this->declare_parameter("frame_rate", 30.0, param_desc);

    // 曝光时间
    param_desc.description = "Exposure time in microseconds";
    double exposure_line_time;
    CameraGetExposureLineTime(h_camera_, &exposure_line_time);
    param_desc.integer_range[0].from_value =
      static_cast<int64_t>(t_capability_.sExposeDesc.uiExposeTimeMin * exposure_line_time);
    param_desc.integer_range[0].to_value =
      static_cast<int64_t>(t_capability_.sExposeDesc.uiExposeTimeMax * exposure_line_time);
    double exposure_time = this->declare_parameter("exposure_time", 5000, param_desc);
    CameraSetExposureTime(h_camera_, static_cast<int>(exposure_time));
    RCLCPP_INFO(this->get_logger(), "Exposure time = %f us", exposure_time);

    // 模拟增益
    param_desc.description = "Analog gain";
    param_desc.integer_range[0].from_value = t_capability_.sExposeDesc.uiAnalogGainMin;
    param_desc.integer_range[0].to_value = t_capability_.sExposeDesc.uiAnalogGainMax;
    int analog_gain;
    CameraGetAnalogGain(h_camera_, &analog_gain);
    analog_gain = this->declare_parameter("analog_gain", analog_gain, param_desc);
    CameraSetAnalogGain(h_camera_, analog_gain);
    RCLCPP_INFO(this->get_logger(), "Analog gain = %d", analog_gain);

    // RGB增益
    CameraGetGain(h_camera_, &r_gain_, &g_gain_, &b_gain_);
    
    // R增益
    param_desc.integer_range[0].from_value = t_capability_.sRgbGainRange.iRGainMin;
    param_desc.integer_range[0].to_value = t_capability_.sRgbGainRange.iRGainMax;
    r_gain_ = this->declare_parameter("rgb_gain.r", r_gain_, param_desc);
    
    // G增益
    param_desc.integer_range[0].from_value = t_capability_.sRgbGainRange.iGGainMin;
    param_desc.integer_range[0].to_value = t_capability_.sRgbGainRange.iGGainMax;
    g_gain_ = this->declare_parameter("rgb_gain.g", g_gain_, param_desc);
    
    // B增益
    param_desc.integer_range[0].from_value = t_capability_.sRgbGainRange.iBGainMin;
    param_desc.integer_range[0].to_value = t_capability_.sRgbGainRange.iBGainMax;
    b_gain_ = this->declare_parameter("rgb_gain.b", b_gain_, param_desc);
    
    CameraSetGain(h_camera_, r_gain_, g_gain_, b_gain_);
    RCLCPP_INFO(this->get_logger(), "RGB Gain: R=%d, G=%d, B=%d", r_gain_, g_gain_, b_gain_);

    // 饱和度
    param_desc.description = "Saturation";
    param_desc.integer_range[0].from_value = t_capability_.sSaturationRange.iMin;
    param_desc.integer_range[0].to_value = t_capability_.sSaturationRange.iMax;
    int saturation;
    CameraGetSaturation(h_camera_, &saturation);
    saturation = this->declare_parameter("saturation", saturation, param_desc);
    CameraSetSaturation(h_camera_, saturation);
    RCLCPP_INFO(this->get_logger(), "Saturation = %d", saturation);

    // Gamma值
    param_desc.integer_range[0].from_value = t_capability_.sGammaRange.iMin;
    param_desc.integer_range[0].to_value = t_capability_.sGammaRange.iMax;
    int gamma;
    CameraGetGamma(h_camera_, &gamma);
    gamma = this->declare_parameter("gamma", gamma, param_desc);
    CameraSetGamma(h_camera_, gamma);
    RCLCPP_INFO(this->get_logger(), "Gamma = %d", gamma);

    // 图像翻转
    flip_image_ = this->declare_parameter("flip_image", false);
    RCLCPP_INFO(this->get_logger(), "Flip image: %s", flip_image_ ? "enabled" : "disabled");
    
    // 设置镜像模式
    {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        // 同时设置水平和垂直镜像
        CameraSetMirror(h_camera_, MIRROR_DIRECTION_HORIZONTAL, flip_image_);
        CameraSetMirror(h_camera_, MIRROR_DIRECTION_VERTICAL, flip_image_);
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    
    // 添加运行状态检查
    if (!running_) {
        result.successful = false;
        result.reason = "Camera is not running";
        return result;
    }
    
    std::lock_guard<std::mutex> lock(camera_mutex_);
    
    for (const auto & param : parameters) {
      try {
        if (param.get_name() == "exposure_time") {
          int status = CameraSetExposureTime(h_camera_, static_cast<int>(param.as_int()));
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set exposure time, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "Exposure time set to %ld us", param.as_int());
          }
        } else if (param.get_name() == "analog_gain") {
          int status = CameraSetAnalogGain(h_camera_, static_cast<int>(param.as_int()));
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set analog gain, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "Analog gain set to %ld", param.as_int());
          }
        } else if (param.get_name() == "rgb_gain.r") {
          r_gain_ = static_cast<int>(param.as_int());
          int status = CameraSetGain(h_camera_, r_gain_, g_gain_, b_gain_);
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set R gain, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "RGB gain R set to %d", r_gain_);
          }
        } else if (param.get_name() == "rgb_gain.g") {
          g_gain_ = static_cast<int>(param.as_int());
          int status = CameraSetGain(h_camera_, r_gain_, g_gain_, b_gain_);
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set G gain, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "RGB gain G set to %d", g_gain_);
          }
        } else if (param.get_name() == "rgb_gain.b") {
          b_gain_ = static_cast<int>(param.as_int());
          int status = CameraSetGain(h_camera_, r_gain_, g_gain_, b_gain_);
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set B gain, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "RGB gain B set to %d", b_gain_);
          }
        } else if (param.get_name() == "saturation") {
          int status = CameraSetSaturation(h_camera_, static_cast<int>(param.as_int()));
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set saturation, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "Saturation set to %ld", param.as_int());
          }
        } else if (param.get_name() == "gamma") {
          int gamma = static_cast<int>(param.as_int());
          int status = CameraSetGamma(h_camera_, gamma);
          if (status != CAMERA_STATUS_SUCCESS) {
            result.successful = false;
            result.reason = "Failed to set gamma, status: " + std::to_string(status);
          } else {
            RCLCPP_INFO(this->get_logger(), "Gamma set to %d", gamma);
          }
        } else if (param.get_name() == "flip_image") {
          flip_image_ = param.as_bool();
          // 同时设置水平和垂直镜像
          CameraSetMirror(h_camera_, MIRROR_DIRECTION_HORIZONTAL, flip_image_);
          CameraSetMirror(h_camera_, MIRROR_DIRECTION_VERTICAL, flip_image_);
          RCLCPP_INFO(this->get_logger(), "Flip image %s", flip_image_ ? "enabled" : "disabled");
        } else if (param.get_name() == "frame_rate") {
          double new_frame_rate = param.as_double();
          if (new_frame_rate > 0 && new_frame_rate <= 60.0) {
            frame_rate_ = new_frame_rate;
            RCLCPP_INFO(this->get_logger(), "Frame rate changed to %.1f Hz", frame_rate_);
          } else {
            result.successful = false;
            result.reason = "Invalid frame rate value: " + std::to_string(new_frame_rate);
          }
        } else {
          RCLCPP_WARN(this->get_logger(), "Unknown parameter: %s", param.get_name().c_str());
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Parameter callback error: %s", e.what());
        result.successful = false;
        result.reason = "Exception: " + std::string(e.what());
      }
    }
    return result;
  }

  int h_camera_ = -1;
  tSdkCameraDevInfo t_camera_enum_list_;
  tSdkCameraCapbility t_capability_;

  image_transport::CameraPublisher camera_pub_;

  // RGB增益
  int r_gain_ = 0, g_gain_ = 0, b_gain_ = 0;

  bool flip_image_ = false;
  double frame_rate_ = 30.0;

  std::string camera_name_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  int fail_count_ = 0;
  std::thread capture_thread_;
  std::mutex camera_mutex_;
  std::atomic<bool> running_{false};

  OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};

}  // namespace mindvision_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(mindvision_camera::MVCameraNode)