#include <fstream>
#include <jsoncpp/json/json.h>
#include <thread>

#include "parameters.hpp"

namespace paacman{


void PaacmanParameters::loadParameters(std::string cfg_file){
  Json::Value root;

  // read root cfg
  std::ifstream ifs(cfg_file);
  ifs >> root;

  // get root of cfg
  std::string cfg_root = root["cfg_root"].asString();

  /* read camera info */
  fs::path camera_info_filename = root["camera_info_filename"].asString();
  camera_info_filename = cfg_root / camera_info_filename;

  Json::Value camera_info_json;
  std::ifstream ifs_camera_info(camera_info_filename);
  ifs_camera_info >> camera_info_json;

  camera_info.width = camera_info_json.get("width", 0).asInt();
  camera_info.height = camera_info_json["height"].asInt();
  if (camera_info_json["K"].size() != 9) my_error("camera_info [K] size is not 9]");
  if (camera_info_json["D"].size() != 4) my_error("camera_info [D] size is not 4");
  vector<float> K_vec, D_vec;
  for(int i = 0; i < (int)camera_info_json["K"].size(); i++){
    K_vec.push_back(camera_info_json["K"][i].asFloat());
  }
  camera_info.K = cv::Mat(3, 3, CV_32F, K_vec.data()).clone();
  for(int i = 0; i < (int)camera_info_json["D"].size(); i++){
    D_vec.push_back(camera_info_json["D"][i].asFloat());
  }
  camera_info.D = cv::Mat(1, 4, CV_32F, D_vec.data()).clone();
  camera_info.new_K = cv::getOptimalNewCameraMatrix(
    camera_info.K, camera_info.D,
    cv::Size(camera_info.width, camera_info.height),
    1
  );

  /* algorithm params */ //this must be initialized before marker_config
  fs::path algo_params_filename = root["algorithm_params_filename"].asString();
  algo_params_filename = cfg_root / algo_params_filename;

  Json::Value algo_params_json;
  std::ifstream ifs_algo_params(algo_params_filename);
  ifs_algo_params >> algo_params_json;

  algo_params.max_available_threads = algo_params_json["max_available_threads"].asInt();
  //get # of available threads of host computer
  if (algo_params.max_available_threads == -1)
    algo_params.max_available_threads = std::thread::hardware_concurrency();
  algo_params.border_cell = algo_params_json["border_cell"].asInt();
  algo_params.step2_labelling_thresh1 = algo_params_json["step2_labelling_thresh1"].asFloat();
  algo_params.step2_labelling_thresh2 = algo_params_json["step2_labelling_thresh2"].asFloat(); 
  algo_params.step2_dist_from_center_rate_error_thresh 
    = algo_params_json["step2_dist_from_center_rate_error_thresh"].asFloat();
  algo_params.step2_right_triangle_rate_error_thresh 
    = algo_params_json["step2_right_triangle_rate_error_thresh"].asFloat();
  algo_params.step2_dist_from_other_to_each_rate_error_thresh 
    = algo_params_json["step2_dist_from_other_to_each_rate_error_thresh"].asFloat();

  algo_params.step2_feature_direction_angle_error_thresh 
    = algo_params_json["step2_feature_direction_angle_error_thresh"].asFloat();
  algo_params.step2_img_and_pnp_yawdiff_error_thresh 
    = algo_params_json["step2_img_and_pnp_yawdiff_error_thresh"].asFloat();
  algo_params.step2_max_reprojection_error_pixel_thresh 
    = algo_params_json["step2_max_reprojection_error_pixel_thresh"].asFloat();

  /* marker config */
  fs::path marker_config_filename = root["marker_config_filename"].asString();
  marker_config_filename = cfg_root / marker_config_filename;

  Json::Value marker_config_json;
  std::ifstream ifs_marker_config(marker_config_filename);
  ifs_marker_config >> marker_config_json;

  Json::Value marker_config_names = marker_config_json["names"];
  for (int i = 0; i < (int)marker_config_names.size(); ++i){
    string name = marker_config_names[i].asString();
    marker_names.push_back(name);

    // keypoint related
    int keypoint_id = marker_config_json[name]["keypoint_id"].asInt();
    const auto& keypoint_descriptor_json = marker_config_json[name]["keypoint_descriptor"];
    vector<int> keypoint_descriptor;
    for (int ii = 0; ii < (int)keypoint_descriptor_json.size(); ++ii){
      keypoint_descriptor.push_back(keypoint_descriptor_json[ii].asInt());
    }

    //marker related
    float keypoint_circle_rad_over_hexagon_rad 
      = marker_config_json[name]["keypoint_circle_rad_over_hexagon_rad"].asFloat();
    float keypoint0_yaw_deg = marker_config_json[name]["keypoint0_yaw_deg"].asFloat();
      
    float hexagon_radius_meter
      = marker_config_json[name]["hexagon_radius_meter"].asFloat();

    marker_configs.emplace(
      name, 
      MarkerConfig(name, 
        keypoint_id, keypoint_descriptor,
        keypoint_circle_rad_over_hexagon_rad, keypoint0_yaw_deg, hexagon_radius_meter,
        STEP2_MIN_RADIUS_PIXELS
        ,true, "/shared_dir/paacman_result/keypoint_patches"
      )
    );
  }

  /* DNN info */
  fs::path dnn_info_filename = root["dnn_info_filename"].asString();
  dnn_info_filename = cfg_root / dnn_info_filename;

  Json::Value dnn_info_json;
  std::ifstream ifs_dnn_info(dnn_info_filename);
  ifs_dnn_info >> dnn_info_json;

  dnn_info.trt_engine_path = dnn_info_json["trt_engine_path"].asString();
  dnn_info.n_desc = dnn_info_json["n_desc"].asInt();
  dnn_info.nms_suppress_dist = dnn_info_json["nms_suppress_dist"].asInt();
  dnn_info.conf_thresh = dnn_info_json["conf_thresh"].asFloat();

  /* debug info */
  fs::path debug_info_filename = root["debug_info_filename"].asString();
  debug_info_filename = cfg_root / debug_info_filename;

  Json::Value debug_info_json;
  std::ifstream ifs_debug_info(debug_info_filename);
  ifs_debug_info >> debug_info_json;

  debug_info.enable_time_log = debug_info_json["enable_time_log"].asBool();
  debug_info.debug_image_dir_root = debug_info_json["debug_image_dir_root"].asString();
  auto debug_image_dirs = debug_info_json["debug_image_dirs"];
  if (debug_image_dirs.size() != (N_STEP+1)) my_error("debug_image_dirs size is not N_STEP");
  debug_info.debug_image_dirs_map[STEP_ZERO]  = debug_image_dirs[0].asString();
  debug_info.debug_image_dirs_map[STEP1]      = debug_image_dirs[1].asString();
  debug_info.debug_image_dirs_map[STEP2]      = debug_image_dirs[2].asString();
  debug_info.clear_debug_image_dirs = debug_info_json["clear_debug_image_dirs"].asBool();

  auto& id0_color_json = debug_info_json["id0_color_cv"];
  debug_info.id0_color = cv::Scalar(id0_color_json[0].asInt(), id0_color_json[1].asInt(), id0_color_json[2].asInt());
  auto& id1_color_json = debug_info_json["id1_color_cv"];
  debug_info.id1_color = cv::Scalar(id1_color_json[0].asInt(), id1_color_json[1].asInt(), id1_color_json[2].asInt());
  auto& id2_color_json = debug_info_json["id2_color_cv"];
  debug_info.id2_color = cv::Scalar(id2_color_json[0].asInt(), id2_color_json[1].asInt(), id2_color_json[2].asInt());
  auto& id3_color_json = debug_info_json["id3_color_cv"];
  debug_info.id3_color = cv::Scalar(id3_color_json[0].asInt(), id3_color_json[1].asInt(), id3_color_json[2].asInt());

  auto& marker_center_color_json = debug_info_json["marker_center_color_cv"];
  debug_info.marker_center_color_cv = 
    cv::Scalar(marker_center_color_json[0].asInt(), marker_center_color_json[1].asInt(), marker_center_color_json[2].asInt());

  debug_info.enclosed_circle_radius = debug_info_json["enclosed_circle_radius"].asInt();
  debug_info.enclosed_circle_thickness = debug_info_json["enclosed_circle_thickness"].asInt();
  debug_info.font_scale = debug_info_json["font_scale"].asFloat();

  debug_info.step1_debug_feature_radius = debug_info_json["step1_debug_feature_radius"].asInt();

  debug_info.enable_debug = debug_info_json["enable_debug"].asBool();
  debug_info.enable_debug_combinatorial_optimization = debug_info_json["enable_debug_combinatorial_optimization"].asBool();
  debug_info.time_log_csv = debug_info_json["time_log_csv"].asString();
  debug_info.time_log_flush_seq = debug_info_json["time_log_flush_seq"].asUInt();
  debug_info.time_log_do_print = debug_info_json["time_log_do_print"].asBool();
}

void PaacmanParameters::printParams(){
  cout << "========================================" << endl;
  cout << "----------------Camera Info-------------" << endl;
  cout << "width: " << camera_info.width << endl;
  cout << "height: " << camera_info.height << endl;
  cout << "K: " << endl;
  cout << camera_info.K << endl;
  cout << "D: " << endl;
  cout << camera_info.D << endl;
  cout << endl;

  cout << "----------------Marker Config-------------" << endl;
  //get keys
  for (const auto& m_name: marker_names){
    const auto& m_config = marker_configs[m_name];
    cout << "* name: " << m_config.name_ << endl;
    
    cout << "  keypoint_id: " << m_config.keypoint_id_ << endl;
    for (const auto& pt3d : m_config.label_to_pts3d){
      cout << "    " << pt3d << endl;
    }

    cout << "  keypoint_circle_rad_over_hexagon_rad: " << m_config.keypoint_circle_rad_over_hexagon_rad_ << endl;
    cout << "  keypoint0_yaw_deg_: " << m_config.keypoint0_yaw_deg_ << endl;
    cout << "  hexagon_radius_meter: " << m_config.hexagon_radius_meter_ << endl;
  }

  cout << "----------------DNN info-------------" << endl;
  cout << "trt_engine_path: " << dnn_info.trt_engine_path << endl;
  cout << "n_desc: " << dnn_info.n_desc << endl;
  cout << "nms_suppress_dist: " << dnn_info.nms_suppress_dist << endl;
  cout << "conf_thresh: " << dnn_info.conf_thresh << endl;

  cout << "--------Algorithm Parameters----------" << endl;
  cout << "max_available_threads: " << algo_params.max_available_threads << endl;
  cout << "border_cell          : " << algo_params.border_cell << endl;
  cout << "step2_labelling_thresh1: " << algo_params.step2_labelling_thresh1 << endl;
  cout << "step2_labelling_thresh2: " << algo_params.step2_labelling_thresh2 << endl;
  cout << "step2_dist_rate_error_thresh: " << algo_params.step2_dist_from_center_rate_error_thresh << endl;
  cout << "step2_right_triangle_rate_error_thresh: " << algo_params.step2_right_triangle_rate_error_thresh << endl;
  cout << "step2_dist_from_other_to_each_rate_error_thresh: " << algo_params.step2_dist_from_other_to_each_rate_error_thresh << endl;
  cout << "step2_feature_direction_angle_error_thresh: " << algo_params.step2_feature_direction_angle_error_thresh << endl;
  cout << "step2_img_and_pnp_yawdiff_error_thresh: " << algo_params.step2_img_and_pnp_yawdiff_error_thresh << endl;
  cout << "step2_max_reprojection_error_pixel_thresh: " << algo_params.step2_max_reprojection_error_pixel_thresh << endl;

  cout << "--------Debug info----------" << endl;
  cout << "* debug info" << endl;
  cout << "  enable_debug: " << debug_info.enable_debug << endl;
  cout << "  enable_debug_combinatorial_optimization: " << debug_info.enable_debug_combinatorial_optimization << endl;
  cout << "  debug_image_dir_root: " << debug_info.debug_image_dir_root << endl;
  cout << "  debug_image_dirs" << endl;
  cout << "    " << debug_info.debug_image_dirs_map[STEP_ZERO] << endl;
  cout << "    " << debug_info.debug_image_dirs_map[STEP1] << endl;
  cout << "    " << debug_info.debug_image_dirs_map[STEP2] << endl;
  cout << "  clear_debug_image_dirs: " << debug_info.clear_debug_image_dirs << endl;
  cout << "  id0_color: " << debug_info.id0_color << endl;
  cout << "  id1_color: " << debug_info.id1_color << endl;
  cout << "  id2_color: " << debug_info.id2_color << endl;
  cout << "  marker_center_color_cv: " << debug_info.marker_center_color_cv << endl;
  cout << "  enclosed_circle_radius: " << debug_info.enclosed_circle_radius << endl;
  cout << "  enclosed_circle_thickness: " << debug_info.enclosed_circle_thickness << endl;
  cout << "  font_scale: " << debug_info.font_scale << endl;
  cout << "  step1_debug_feature_radius: " << debug_info.step1_debug_feature_radius << endl;

  cout << "* time log" << endl;
  cout << "  enable_time_log: " << debug_info.enable_time_log << endl;
  cout << "  time_log_csv: " << debug_info.time_log_csv << endl;
  cout << "  time_log_flush_seq: " << debug_info.time_log_flush_seq << endl;
  

  cout << "========================================" << endl << endl;
}


} // namespace paacman
