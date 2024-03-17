/**
 * date: 2024-3-17
 * author: lijun
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <queue>
using EigenContour = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;
using CvPoints = std::vector<cv::Point2f>;
using EVec2f = Eigen::Vector2f;

// 机器人速度窗口
typedef struct win
{
  float min_velocity;
  float max_velocity;
  float min_yawrate;
  float max_yawrate;
} Window;

// 机器人状态
typedef struct state
{
  float x = 0;           // robot position x
  float y = 0;           // robot posiiton y
  float yaw = 0;         // robot orientation yaw
  float linear_vel = 0;  // robot linear velocity
  float angular_vel = 0; // robot angular velocity
} State;

// 记录机器人位置
typedef struct pose
{
  float x;
  float y;
  float angle;
} Pose2f;

// 记录机器人速度
typedef struct odom
{
  float v;
  float w;
} Odom;

class DWAPlanner
{
public:
  DWAPlanner();
  ~DWAPlanner(){};
  void setGoal(const Eigen::Vector3f &goal);
  void setPath(EigenContour &path);
  bool plan(Pose2f &pose, Odom &odom);
  EigenContour lineSample(const EVec2f &start, const EVec2f &end, const float delta);
  void getTrajectory(std::vector<std::vector<State>> &all_tra, std::vector<State> &best_tra);

private:
  void setState(Pose2f &pose, const float &v, const float &w);

  void dwaPlanning(const Window &window, const std::deque<Eigen::Vector3f> &pruned_plan, std::vector<State> &best_traj);

  void updateMotion(State &state, const float &v, const float &w,
                    const float &dt);
  float calcSpeedCost(const std::vector<State> &traj, const float target_vel);

  float calcGoalCost(const std::vector<State> &traj, const Eigen::Vector3f &goal);

  float calcPathCost(const std::vector<State> &traj, const std::deque<Eigen::Vector3f> &pruned_plan);

  void getLocalPath(Pose2f &pose, Odom &odom, std::deque<Eigen::Vector3f> &pruned_plan);

  void prunePath(Pose2f &pose);

  float angleNormalize(float theta);

  // kdtree::KDTree* mpKDTree;
  Eigen::Vector3f goal_;
  std::deque<Eigen::Vector3f> path_;
  State state_;
  Window window_;

  // 控制频率
  float hz_;
  float target_vel_;
  float max_linear_vel_;
  float min_linear_vel_;
  float max_angular_vel_;
  float min_angular_vel_;
  float max_linear_acc_;
  float max_angular_acc_;
  float predict_time_;
  float dt_;

  // 采样分辨率
  float n_vsamples_;
  float n_wsamples_;

  // 代价因子
  float goal_cost_gain_;
  float path_cost_gain_;
  float speed_cost_gain_;
  float obstacle_cost_gain_;

  // 转弯角度阈值
  float prune_angle_;
  // 存放所有的轨迹
  std::vector<std::vector<State>> all_traj_;
  // 存放最优轨迹；
  std::vector<State> best_traj_;
};
