#pragma once

#include <mutex>

#include "common.hpp"
#include "parameters.hpp"
#include "debuggingInfo.hpp"
#include "detectedMarker.hpp"
#include "tensorrt10Manager.hpp"


namespace paacman{

struct STEP_OUT{
  int step_type;
  bool success;
};
struct STEP1_OUT: public STEP_OUT{
  // to step2
  vector<DetectedFeature> detected_features;

  // used while processing current step
  vector<vector<float>> net_output_vectors;
};

struct STEP2_OUT: public STEP_OUT{
  map<int, bool> id_to_is_valid;
  map<int, vector<DetectedFeature>> id_to_features;
  map<int, vector<float>> id_to_undistorted_pts_distmap; // 1d array but 
  // handled as 2d array with last member == width(height)
  float getDistanceBtw2UndistoredPts(int id, int left, int right) const {
    const int distmap_width = id_to_undistorted_pts_distmap.at(id).back();
    return id_to_undistorted_pts_distmap.at(id)[left * distmap_width + right];
  }

  map<int, DetectedMarker> id_to_detected_markers;
};


struct CheckInfo {
  virtual ~CheckInfo() = 0;
};

struct EquilateralShapeCheckInfo: public CheckInfo{
  // regular shape check
  PaacmanParameters* params_ptr;
  map<int, vector<DetectedFeature>>* id_to_features_ptr;
  const vector<vector<int>>* combinations_helper_vec2d_ptr;
  int helper_start_idx;
  int helper_end_idx;
  int id;
};

struct CheckDetail {
  virtual void clear() = 0;
  virtual DetectedMarker toDetectedMarker(int id) const = 0;
};

struct LabellingAndDirectionCheckDetail: public CheckDetail{
  bool success;
  struct LabellingDetail{
    vector<int> pseudo_labels;
    
    cv::Point2f hexagon_center_undistorted;
    vector<cv::Point2f> pts_clockwise_distorted;
    vector<cv::Point2f> pts_clockwise_undistorted;
  } labelling_detail;

  struct DirectionCheckDetail{
    bool success;
    float min_radius_pixel_img;
    float max_radius_pixel_img;
    vector<float> directions_clockwise_degree;
    vector<int> labels;

    cv::Mat tvec, rvec;
    double yawdiff_error;
    double reprojection_error;
    LabellingDetail* l_detail_ptr;
  } direction_check_detail;

  void clear();
  DetectedMarker toDetectedMarker(int id) const;
};

class Paacman{
public:
  Paacman(string cfg_path);

  vector<DetectedFeature> detectKeypoints(const cv::Mat &img);
  bool detectMarkers(const cv::Mat &img, map<int, DetectedMarker> &id_to_detected_markers);
  bool visualizeDetectedMarkers
    (const cv::Mat& in, cv::Mat& out, const map<int, DetectedMarker> &id_to_detected_markers,
    bool print_on_failure = false);
  // utilities
  const PaacmanParameters& getParams();
  Paacman& setTimeLogging(bool enable);
  Paacman& setDebugging(bool enable);

private:
  
  // initialize
  void initPaacman();

  // run utilities
  void initUtilities();
  void runUtilities(int step, const cv::Mat* img_ptr, 
    STEP_OUT* step_out_ptr = nullptr, bool print_separator=false);

  bool step1_network_inference(
    const cv::Mat& img, 
    STEP1_OUT& step1_out_ptr
  );

  bool step2_shape_check(
    const cv::Mat& img,
    const STEP1_OUT& step1_out,
    STEP2_OUT& step2_out
  );

  void step2_check_if_regular( // called internally by step2_shape_check()
    const cv::Mat* img_ptr,
    const CheckInfo* regular_shape_check_info_ptr,
    CheckDetail* regular_shape_detail_ptr
  );
  void step2_check_if_equilateral( // called internally by step2_shape_check()
    const cv::Mat* img_ptr,
    const CheckInfo* h_and_dir_info_ptr,
    CheckDetail* h_and_dir_detail_ptr
  );
  std::mutex step2_check_if_regular_mtx;
  bool found_marker;
  
  // parameters, configs, infos
  PaacmanParameters params;

  // step1 members
  Tensorrt10Manager* trt_mgr_ptr;

  // utilities
  bool enable_time_logging, enable_debugging;

  TimeLogger time_logger;
  PaacmanDebugDetail debug_detail;
  int seq;

  /* step2 parameters */
  EquilateralShapeCheckInfo info_ary[MAX_ALLOWED_THREADS];
  LabellingAndDirectionCheckDetail detail_ary[MAX_ALLOWED_THREADS]; // must be lower then n_max_thread
  vector<vector<vector<int>>> combination_helper_vec3d; 
  // ex: [n_feature_cand][combination_idx] = {0, 1, 3, 4}

  // regular shape check
  map<int, vector<vector<int>>> label_type_to_pseudo_labels;
  map<int, vector<vector<int>>> label_type_to_right_triangle;
  map<int, vector<int>> label_type_to_right_triangle_other_idx;
  map<int, vector<float>> label_type_to_inverse_area;
  vector<float> clock_dist_diff_to_length_3d_ratio;

};
} // namespace paacman
