/**
 * date: 2024-3-17
 * author: lijun
 */

#include "../include/calculate_param_dwa.hpp"
#include <string>

DWAPlanner::DWAPlanner()
{
  // 控制频率
  hz_ = 20;
  dt_ = 1.0 / hz_;

  min_linear_vel_ = 0.15;
  max_linear_vel_ = 0.25;
  max_linear_acc_ = 1.0;
  max_angular_vel_ = 1.2;
  min_angular_vel_ = 0.5;
  max_angular_acc_ = 2.0;

  // 预测时间
  predict_time_ = 4;
  // 采样分辨率
  n_vsamples_ = 20;
  n_wsamples_ = 50;
  // 代价因子
  path_cost_gain_ = 1.0;
  obstacle_cost_gain_ = 0.0;
  goal_cost_gain_ = 0.0;
  speed_cost_gain_ = 0.0;
  window_.min_velocity = min_linear_vel_;
  window_.max_velocity = max_linear_vel_;
  window_.min_yawrate = -max_angular_vel_;
  window_.max_yawrate = max_angular_vel_;
  prune_angle_ = 45 * M_PI / 180;
}

bool DWAPlanner::plan(Pose2f &pose, Odom &odom)
{
  // update current state
  setState(pose, odom.v, odom.w);
  // prune plan
  prunePath(pose);
  if (path_.size() <= 1)
  {
    return false;
  }
  // get local plan
  std::deque<Eigen::Vector3f> local_path;
  getLocalPath(pose, odom, local_path);

  // 清除所有的轨迹
  best_traj_.clear();
  all_traj_.clear();
  dwaPlanning(window_, local_path, best_traj_);
  return true;
}

void DWAPlanner::setGoal(const Eigen::Vector3f &goal) { goal_ = goal; }

void DWAPlanner::setPath(EigenContour &path)
{
  while (!path_.empty())
    path_.pop_front();
  float yaw{0.0};
  yaw = atan2(path[1].y() - path[0].y(), path[1].x() - path[0].x());
  Eigen::Vector2f prev_point = path[0];
  path_.push_back(Eigen::Vector3f(path[0].x(), path[0].y(), yaw));
  for (size_t i = 1; i < path.size(); ++i)
  {
    yaw = atan2(path[i].y() - prev_point.y(), path[i].x() - prev_point.x());
    path_.push_back(Eigen::Vector3f(path[i].x(), path[i].y(), yaw));
    prev_point = path[i];
  }
  goal_ = path_.back();
}

void DWAPlanner::dwaPlanning(const Window &dynamic_window, const std::deque<Eigen::Vector3f> &local_plan, std::vector<State> &best_traj)
{
  float min_cost = 1e9;
  float vel_resolution = (dynamic_window.max_velocity - dynamic_window.min_velocity) / n_vsamples_;
  float w_resolution = (dynamic_window.max_yawrate - dynamic_window.min_yawrate) / n_wsamples_;
  int pruned_plan_size = local_plan.size();
  float dt = (float)predict_time_ / pruned_plan_size;
  float goal_cost, speed_cost, path_cost, obstacle_cost, final_cost;
  Eigen::Vector3f local_goal = local_plan.back();
  for (float v = dynamic_window.min_velocity; v <= dynamic_window.max_velocity; v += vel_resolution)
  {
    for (float w = dynamic_window.min_yawrate; w <= dynamic_window.max_yawrate; w += w_resolution)
    {
      std::vector<State> traj;
      State local_state = state_;
      traj.push_back(local_state);
      for (int i = 0; i < pruned_plan_size - 1; i++)
      {
        updateMotion(local_state, v, w, dt);
        traj.emplace_back(local_state);
      }
      // 存放当前所有的路径轨迹
      all_traj_.push_back(traj);
      goal_cost = calcGoalCost(traj, local_goal);
      speed_cost = calcSpeedCost(traj, max_linear_vel_);
      path_cost = calcPathCost(traj, local_plan);
      obstacle_cost = 0.0;
      final_cost = goal_cost_gain_ * goal_cost + speed_cost_gain_ * speed_cost + obstacle_cost_gain_ * obstacle_cost + path_cost_gain_ * path_cost;
      if (final_cost <= min_cost)
      {
        min_cost = final_cost;
        best_traj = traj;
      }
    }
  }
}

void DWAPlanner::setState(Pose2f &pose, const float &v, const float &w)
{
  state_.x = pose.x;
  state_.y = pose.y;
  state_.yaw = pose.angle;
  state_.linear_vel = v;
  state_.angular_vel = w;
}

void DWAPlanner::prunePath(Pose2f &pose)
{
  float min_dist = 1e6;
  int min_index = 0;
  float x = pose.x;
  float y = pose.y;
  for (size_t i = 0; i < path_.size(); i++)
  {
    float dist = (path_[i].x() - x) * (path_[i].x() - x) +
                 (path_[i].y() - y) * (path_[i].y() - y);
    if (dist < min_dist)
    {
      min_dist = dist;
      min_index = i;
    }
  }
  for (int i = 0; i < min_index; ++i)
  {
    path_.pop_front();
  }
}

void DWAPlanner::getLocalPath(Pose2f &pose, Odom &odom,
                              std::deque<Eigen::Vector3f> &local_path)
{
  float v = odom.v + min_linear_vel_;
  if (v > max_linear_vel_)
    v = max_linear_vel_;
  if (v < min_linear_vel_)
    v = min_linear_vel_;
  float window_r = v * predict_time_;
  Eigen::Vector2f point(pose.x, pose.y);
  float angle = pose.angle;
  local_path.push_back(Eigen::Vector3f(point.x(), point.y(), angle));
  for (size_t i = 0; i < path_.size(); i++)
  {
    local_path.push_back(path_[i]);
    Eigen::Vector2f front_point = local_path.back().segment(0, 2);
    float dist = (front_point - point).norm();

    float bias_angle = angleNormalize(pose.angle - atan2(front_point.y() - point.y(), front_point.x() - point.x()));
    if (dist > window_r || (dist > min_linear_vel_ * predict_time_ && fabs(bias_angle) > prune_angle_))
    {
      break;
    }
  }
}

void DWAPlanner::updateMotion(State &state, const float &v, const float &w, const float &dt)
{
  float unit_dt = 0.002;
  int dt_size = dt / unit_dt;
  for (int i = 0; i < dt_size; i++)
  {
    state.yaw += w * unit_dt;
    state.x += v * std::cos(state.yaw) * unit_dt;
    state.y += v * std::sin(state.yaw) * unit_dt;
  }
  state.linear_vel = v;
  state.angular_vel = w;
}

float DWAPlanner::calcSpeedCost(const std::vector<State> &traj,
                                const float target_vel)
{
  float cost = fabs(target_vel - traj.back().linear_vel);
  return cost;
}

float DWAPlanner::calcGoalCost(const std::vector<State> &traj,
                               const Eigen::Vector3f &goal)
{
  Eigen::Vector3f end_pose(traj.back().x, traj.back().y, traj.back().yaw);
  return (end_pose.segment(0, 2) - goal.segment(0, 2)).norm();
}

float DWAPlanner::calcPathCost(const std::vector<State> &dst_traj,
                               const std::deque<Eigen::Vector3f> &src_traj)
{

  float angular_error = 0.0, linear_error = 0.0;
  std::vector<float> src_start_point(2, 0);
  std::vector<float> dst_start_point(2, 0);
  int src_traj_size = src_traj.size();
  int dst_traj_size = dst_traj.size();
  for (int i = 0; i < src_traj_size; i++)
  {
    float dx = dst_traj[i].x - src_traj[i].x();
    float dy = dst_traj[i].y - src_traj[i].y();
    if (i != 0)
    {
      float ddx = dst_traj[i].x - dst_traj[i - 1].x;
      float ddy = dst_traj[i].y - dst_traj[i - 1].y;
      float sdx = src_traj[i].x() - src_traj[i - 1].x();
      float sdy = src_traj[i].y() - src_traj[i - 1].y();
      float d_angle = fabs(angleNormalize(atan2(ddy, ddx) - atan2(sdy, sdx)));
      angular_error += d_angle;
    }
    linear_error += std::sqrt(dx * dx + dy * dy);
  }
  return linear_error + angular_error;
}

float DWAPlanner::angleNormalize(float theta)
{
  if (theta >= -M_PI && theta < M_PI)
    return theta;
  int multiplier = (int)(theta / (2 * M_PI));
  theta = theta - multiplier * 2 * M_PI;
  if (theta >= M_PI)
    theta -= 2 * M_PI;
  if (theta < -M_PI)
    theta += 2 * M_PI;
  return theta;
}

EigenContour DWAPlanner::lineSample(const EVec2f &start, const EVec2f &end, const float delta)
{
  EigenContour path;
  path.emplace_back(start);
  if ((start - end).norm() <= delta)
  {
    path.emplace_back(end);
    return path;
  }
  EVec2f point = start;
  float theta = atan2(end.y() - start.y(), end.x() - start.x());
  float c = cos(theta);
  float s = sin(theta);
  float length = (start - end).norm();
  while (true)
  {
    point += EVec2f(delta * c, delta * s);
    float dist = (point - start).norm();
    if (dist >= length)
      break;
    path.emplace_back(point);
  }
  path.emplace_back(end);
  return path;
}

void DWAPlanner::getTrajectory(std::vector<std::vector<State>> &all_tra, std::vector<State> &best_tra)
{
  all_tra = all_traj_;
  best_tra = best_traj_;
}