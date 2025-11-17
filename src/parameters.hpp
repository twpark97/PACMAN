#pragma once

#include "common.hpp"
#include <bitset>
#include <opencv2/opencv.hpp>

namespace paacman{
  


struct CameraInfo{
  int width;
  int height;
  cv::Mat K, D;
  cv::Mat new_K; // K after undistortion
};

struct AlgoParams{
  // general parameters
  int max_available_threads;

  // step1 params
  int border_cell; // features near the border will be suppressed
                    // near the border == border_cell*8 pixel
 
  // step2 params for shape check
  float step2_labelling_thresh1;
  float step2_labelling_thresh2;
  float step2_dist_from_center_rate_error_thresh; // percentage btw idea and calculated
  float step2_right_triangle_rate_error_thresh; // percentage btw idea and calculated
  float step2_dist_from_other_to_each_rate_error_thresh; // percentage

  // step2 params for direction check
  float step2_feature_direction_angle_error_thresh;
  float step2_img_and_pnp_yawdiff_error_thresh;
  float step2_max_reprojection_error_pixel_thresh;
};

struct DnnInfo{
  string trt_engine_path; // note that this is different from original Superpoint model,
  // since it contains some postprocessing computations(e.g. softmax)
  int n_desc; //# of descriptors that network trained
  int nms_suppress_dist; // pixels near the high confidence feature will be suppressed
  float conf_thresh; // confidence threshold for feature
};

template <int rad> struct rad_to_perimeter_length{enum {value=0};};
template <> struct rad_to_perimeter_length<3> {enum {value=16};};
template <> struct rad_to_perimeter_length<5> {enum {value=32};};
template <> struct rad_to_perimeter_length<7> {enum {value=48};};
template <> struct rad_to_perimeter_length<9> {enum {value=64};};
template <> struct rad_to_perimeter_length<11> {enum {value=72};};
template <> struct rad_to_perimeter_length<13> {enum {value=88};};
template <> struct rad_to_perimeter_length<15> {enum {value=112};};
template <> struct rad_to_perimeter_length<17> {enum {value=128};};
template <> struct rad_to_perimeter_length<19> {enum {value=128};};
template <> struct rad_to_perimeter_length<21> {enum {value=152};};
template <> struct rad_to_perimeter_length<23> {enum {value=156};};

typedef std::bitset<rad_to_perimeter_length<STEP2_MIN_RADIUS_PIXELS[0]>::value> 
  keypoint_descriptor_type0_t;
typedef std::bitset<rad_to_perimeter_length<STEP2_MIN_RADIUS_PIXELS[1]>::value> 
  keypoint_descriptor_type1_t;
typedef std::bitset<rad_to_perimeter_length<STEP2_MIN_RADIUS_PIXELS[2]>::value> 
  keypoint_descriptor_type2_t;

struct KeypointBinaryDescriptorVec{
  vector<keypoint_descriptor_type0_t> 
    keypoint_descriptor_binary_min_radius_pixel_shifted0_;
  vector<keypoint_descriptor_type1_t> 
    keypoint_descriptor_binary_min_radius_pixel_shifted1_;
  vector<keypoint_descriptor_type2_t> 
    keypoint_descriptor_binary_min_radius_pixel_shifted2_;

  auto& getBitsetVec0(){
    return keypoint_descriptor_binary_min_radius_pixel_shifted0_;
  }
  auto& getBitsetVec1(){
    return keypoint_descriptor_binary_min_radius_pixel_shifted1_;
  }
  auto& getBitsetVec2(){
    return keypoint_descriptor_binary_min_radius_pixel_shifted2_;
  }
};

struct KeypointBinaryDescriptors{
  keypoint_descriptor_type0_t keypoint_descriptor_binary_min_radius_pixel0;
  keypoint_descriptor_type1_t keypoint_descriptor_binary_min_radius_pixel1;
  keypoint_descriptor_type2_t keypoint_descriptor_binary_min_radius_pixel2;

  void reset(){
    keypoint_descriptor_binary_min_radius_pixel0.reset();
    keypoint_descriptor_binary_min_radius_pixel1.reset();
    keypoint_descriptor_binary_min_radius_pixel2.reset();
  }

  void set(int rad_idx, int idx, bool val){
    if (rad_idx == 0){
      keypoint_descriptor_binary_min_radius_pixel0.set(idx, val);
    }
    else if (rad_idx == 1){
      keypoint_descriptor_binary_min_radius_pixel1.set(idx, val);
    }
    else if (rad_idx == 2){
      keypoint_descriptor_binary_min_radius_pixel2.set(idx, val);
    }
    else{
      my_error("not implemented for rad > 23 in KeypointBinaryDescriptor::set");
    }
  }

  bool getBit(int idx_rad, int idx) const{
    if (idx_rad == 0){
      return keypoint_descriptor_binary_min_radius_pixel0[idx];
    }
    else if (idx_rad == 1){
      return keypoint_descriptor_binary_min_radius_pixel1[idx];
    }
    else if (idx_rad == 2){
      return keypoint_descriptor_binary_min_radius_pixel2[idx];
    }
    else{
      my_error("not implemented for rad > 23 in KeypointBinaryDescriptor::operator[]");
    }
    return false; //dead code
  }
  size_t size(int idx_rad) const{
    if (idx_rad == 0){
      return keypoint_descriptor_binary_min_radius_pixel0.size();
    }
    else if (idx_rad == 1){
      return keypoint_descriptor_binary_min_radius_pixel1.size();
    }
    else if (idx_rad == 2){
      return keypoint_descriptor_binary_min_radius_pixel2.size();
    }
    else{
      my_error("not implemented for rad > 23 in KeypointBinaryDescriptor::size");
    }
    return 0; //dead code
  }
  
  auto& getDescriptor0(){
    return keypoint_descriptor_binary_min_radius_pixel0;
  }
  auto& getDescriptor1(){
    return keypoint_descriptor_binary_min_radius_pixel1;
  }
  auto& getDescriptor2(){
    return keypoint_descriptor_binary_min_radius_pixel2;
  }
  const auto& getDescriptor0() const {
    return keypoint_descriptor_binary_min_radius_pixel0;
  }
  const auto& getDescriptor1() const {
    return keypoint_descriptor_binary_min_radius_pixel1;
  }
  const auto& getDescriptor2() const {
    return keypoint_descriptor_binary_min_radius_pixel2;
  }
};

struct MarkerConfig{
public:
  string name_;
  //keypoint related
  int keypoint_id_;
  KeypointBinaryDescriptors keypoint_binary_descriptors;
  KeypointBinaryDescriptorVec keypoint_binary_descriptor_shifted;
  //marker related
  float keypoint_circle_rad_over_hexagon_rad_;
  float keypoint0_yaw_deg_;
  float hexagon_radius_meter_;
  // computed in constructor
  vector<vector<cv::Point2i>> keypoint_descriptor_dxdys_vec_;
  vector<vector<float>> dedicated_angles_vec;
  vector<cv::Point3f> label_to_pts3d;
  float black_area_ratio_;
  float white_area_ratio_;

public:
  MarkerConfig() = default;
  MarkerConfig(
    string name,
    //keypoint related
    int keypoint_id,
    vector<int> keypoint_descriptor_simple, 
    //marker related
    float keypoint_circle_rad_over_hexagon_rad,
    float keypoint0_yaw_deg,
    float hexagon_radius_meter,
    // predefined
    const int* min_radius_pixel_img,
    // optional
    bool check_binary_descriptor = false,
    string binary_descriptor_save_dir = ""
  ):
    name_(name),
    keypoint_id_(keypoint_id),
    keypoint_circle_rad_over_hexagon_rad_(keypoint_circle_rad_over_hexagon_rad),
    keypoint0_yaw_deg_(keypoint0_yaw_deg),
    hexagon_radius_meter_(hexagon_radius_meter),
    label_to_pts3d(0),
    black_area_ratio_(0),
    white_area_ratio_(0)
  {
    for (int ii = 0 ; ii < N_KEYPOINT_IN_MAKRER; ++ii){
      float angle = keypoint0_yaw_deg / 180.0 * M_PI - 2 * M_PI * ii / N_KEYPOINT_IN_MAKRER;
      float x = hexagon_radius_meter * cos(angle);
      float y = hexagon_radius_meter * sin(angle);
      label_to_pts3d.push_back(cv::Point3f(x, y, 0));
    }
    // black and whte area ratio
    int n_length = (size_t)keypoint_descriptor_simple.size();
    int n_black = 0;
    for (int ii = 0 ; ii < n_length; ++ii){
      if (keypoint_descriptor_simple[ii] == 0)
        n_black++;
    }
    black_area_ratio_ = (float)n_black / n_length;
    white_area_ratio_ = 1 - black_area_ratio_;

    // 90 degree and its multiplicants - forcely expand smaller # of colors
    bool target_bit_to_expand = white_area_ratio_ > black_area_ratio_ ? false: true;

    // get xy points on circle with radius min_radius_pixel_img
    keypoint_descriptor_dxdys_vec_.resize(3); 
    dedicated_angles_vec.resize(3);
    for (int idx_rad = 0 ; idx_rad < 3; ++idx_rad)
    {
      int radius = min_radius_pixel_img[idx_rad];
      if (radius % 2 != 1)
        my_error("min_radius_pixel_img should be odd");
 
      // create binary descriptor considering min_radius_pixel_img
      for (int deg = 0; deg < 360; deg++){
        float rad = deg * M_PI / 180;
        cv::Point2i xy;
        xy.x = (radius-1) * cos(rad);
        xy.y = (radius-1) * sin(rad);
        // insert only unique points  
        if (!keypoint_descriptor_dxdys_vec_[idx_rad].size()){
          keypoint_descriptor_dxdys_vec_[idx_rad].push_back(xy);

          float deg = std::atan2(xy.y, xy.x)*180/M_PI;
          dedicated_angles_vec[idx_rad].push_back(deg);
        }
        else{
          const auto prev_pt = keypoint_descriptor_dxdys_vec_[idx_rad].back();
          if (prev_pt.x != xy.x || prev_pt.y != xy.y){
            keypoint_descriptor_dxdys_vec_[idx_rad].push_back(xy);

            float deg = std::atan2(xy.y, xy.x)*180/M_PI;
            dedicated_angles_vec[idx_rad].push_back(deg);
          }
        }
      }
    }

    for (int idx_rad = 0 ; idx_rad < 3; ++idx_rad)
    {
      int rad = min_radius_pixel_img[idx_rad];
      if (rad % 2 != 1)
        my_error("min_radius_pixel_img should be odd");
 
      // create binary descriptor considering min_radius_pixel_img
      // create keypoint_descriptor_binary_min_radius_pixel
      float given_deg_step = 360.f / keypoint_descriptor_simple.size();
      int bit_idx = 0;
      for (auto [x, y]: keypoint_descriptor_dxdys_vec_[idx_rad])
      {
        float angle = atan2(y, x);
        if (angle < 0)
          angle += 2 * M_PI;
        float angle_deg = angle * 180 / M_PI;

        int idx = angle_deg / given_deg_step;

        keypoint_binary_descriptors.set(
          idx_rad, bit_idx++, keypoint_descriptor_simple[idx] == 1);

      }
      int n_descriptor_length = keypoint_binary_descriptors.size(idx_rad);
     
      int angle_90 = n_descriptor_length / 4;
      
      for (int factor = 1; factor <= 4; ++factor){
        if (keypoint_binary_descriptors.getBit(idx_rad, angle_90*factor - 2) 
          == target_bit_to_expand)
        {
          keypoint_binary_descriptors.set(
            idx_rad, angle_90*factor-1, target_bit_to_expand);
          int idx = angle_90*factor % n_descriptor_length;
          keypoint_binary_descriptors.set(idx_rad, idx, target_bit_to_expand);
        }
      }

      for (int factor = 1; factor <= 4; ++factor){
        // calculate shifted angle ro estimate orientation of rotated keypoints
        if (idx_rad == 0){
          auto& target_descriptor = keypoint_binary_descriptors.getDescriptor0();
          auto& target_descriptor_shifted = 
            keypoint_binary_descriptor_shifted.getBitsetVec0();
          for (int idx_desc = 0; idx_desc < n_descriptor_length; ++idx_desc)
          {
            auto&& res = 
              (target_descriptor << idx_desc) |
              (target_descriptor >> (n_descriptor_length - idx_desc)
            );
            target_descriptor_shifted.emplace_back(res);
          }   
        }
        else if (idx_rad == 1){
          auto& target_descriptor = keypoint_binary_descriptors.getDescriptor1();
          auto& target_descriptor_shifted 
            = keypoint_binary_descriptor_shifted.getBitsetVec1();
          for (int idx_desc = 0; idx_desc < n_descriptor_length; ++idx_desc)
          {
            auto&& res = 
              (target_descriptor << idx_desc) |
              (target_descriptor >> (n_descriptor_length - idx_desc)
            );
            target_descriptor_shifted.emplace_back(res);
          }   
        }
        else if (idx_rad == 2){
          auto& target_descriptor = keypoint_binary_descriptors.getDescriptor2();
          auto& target_descriptor_shifted 
            = keypoint_binary_descriptor_shifted.getBitsetVec2();
          for (int idx_desc = 0; idx_desc < n_descriptor_length; ++idx_desc)
          {
            auto&& res = 
              (target_descriptor << idx_desc) |
              (target_descriptor >> (n_descriptor_length - idx_desc)
            );
            target_descriptor_shifted.emplace_back(res);
          }   
        }
      }
      // draw perimeter descriptor on patch
      if (check_binary_descriptor)
      {
        cv::Mat patch = cv::Mat::ones(2 * rad - 1, 2 * rad - 1, CV_8UC1);
        patch *= 128;
        for (size_t xy_idx = 0; xy_idx < keypoint_descriptor_dxdys_vec_[idx_rad].size(); ++xy_idx){
          int x = keypoint_descriptor_dxdys_vec_[idx_rad][xy_idx].x;
          int y = keypoint_descriptor_dxdys_vec_[idx_rad][xy_idx].y;
          x += rad-1;
          y += rad-1;
          int val = keypoint_binary_descriptors.getBit(idx_rad, xy_idx) * 255;

          patch.at<uchar>(y, x) = val;
        }
        cv::imwrite(binary_descriptor_save_dir + "/" + name 
          + "_keypoint_patch" + std::to_string(idx_rad) + ".png", patch);
      }
    }// idx_rad
  }//function
}; //struct


struct DebugInfo{
  bool enable_debug;
  bool enable_debug_combinatorial_optimization;
  string debug_image_dir_root;
  std::map<int, string> debug_image_dirs_map;
  bool clear_debug_image_dirs;

  cv::Scalar id0_color;
  cv::Scalar id1_color;
  cv::Scalar id2_color;
  cv::Scalar id3_color;
  cv::Scalar marker_center_color_cv;

  int enclosed_circle_radius;
  int enclosed_circle_thickness;
  float font_scale;

  int step1_debug_feature_radius;

  bool enable_time_log;
  string time_log_csv;
  unsigned int time_log_flush_seq;
  bool time_log_do_print;
  cv::Scalar id_to_color(int id) const{

  switch (id){
  case -1: return marker_center_color_cv;
  case 0: return id0_color;
  case 1: return id1_color;
  case 2: return id2_color;
  case 3: return id3_color;
  default: my_error("not implemented for id > 3 in id_to_color");
  }
  
    return cv::Scalar(0,0,0); //dead code
  }
};

using MarkerConfigsType = std::map<string, MarkerConfig>; // id, config

class PaacmanParameters{
public:
  void loadParameters(std::string cfg_file);
  void printParams();

  inline const CameraInfo& getCameraInfo() const { return camera_info; }
  inline const AlgoParams& getAlgoParams() const { return algo_params; }
  inline const DnnInfo& getDnnInfo() const { return dnn_info; }
  inline const MarkerConfigsType& getMarkerConfigs() const { return marker_configs; }
  inline const vector<string>& getMarkerNames() const { return marker_names;}
  inline const DebugInfo& getDebugInfo() const { return debug_info; }

private:
  CameraInfo camera_info;
  AlgoParams algo_params;
  DnnInfo dnn_info;
  MarkerConfigsType marker_configs;
  vector<string> marker_names;
  DebugInfo debug_info;
};

} // namespace paacman
