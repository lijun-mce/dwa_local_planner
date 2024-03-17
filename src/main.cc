/**
 * date: 2024-3-17
 * author: lijun
 * 注释：作者这里有一个问题，把每次计算得到的最优采样轨迹作为了实际跟踪路径
 *       实际上应该根据具体控制频率与计算得到的最优vw去进行推导，所以仅供演示
*/
#include "../include/calculate_param_dwa.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
using EigenContour = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;

int main()
{
  // 将路径数据读进来
  ifstream ifs;
  ifs.open("../data/path_points.txt", ios::in);
  std::vector<cv::Point> path_points;
  // 用于存放全局路径
  EigenContour path_points_world;
  if (ifs.is_open())
  {
    char k;
    cv::Point path_point;
    while (ifs >> path_point.x >> k >> path_point.y)
    {
      path_points.push_back(path_point);
      path_points_world.push_back(
          Eigen::Vector2f(path_point.x * 0.05, path_point.y * 0.05));
    }
  }
  ifs.close();
  
  cv::Mat cv_map = cv::Mat(352, 336, CV_8UC1, cv::Scalar(255));
  for (size_t i = 0; i < path_points.size(); ++i)
  {
    if (i + 1 < path_points.size())
      cv::line(cv_map, path_points[i], path_points[i + 1], cv::Scalar(0), 1);
  }
  imwrite("../test_result/cv_map.png", cv_map);
  // 将二值图转换为彩色图片
  cv::Mat color_map;
  cv::cvtColor(cv_map, color_map, CV_GRAY2BGR);

  DWAPlanner dwa_test;
  EigenContour detail_path;
  for (int i = 0; i < path_points_world.size() - 1; i++)
  {
    EigenContour tmp_path = dwa_test.lineSample(path_points_world[i],
                                                path_points_world[i + 1], 0.05);
    for (int j = 0; j < tmp_path.size() - 1; j++)
    {
      detail_path.emplace_back(tmp_path[j]);
    }
  }
  detail_path.emplace_back(path_points_world.back());
  dwa_test.setPath(detail_path);

  // 设置机器人当前位姿
  Pose2f pose_now;
  pose_now.x = detail_path[0][0];
  pose_now.y = detail_path[0][1];
  pose_now.angle = atan2(detail_path[2][1] - detail_path[0][1],
                         detail_path[2][0] - detail_path[0][0]);
  // 设置机器人当前odom速度
  Odom odom_now;
  odom_now.v = 0;
  odom_now.w = 0;
  while (1)
  {
    if (!dwa_test.plan(pose_now, odom_now))
    {
      cout << "finish planning!!!" << endl;
      break;
    }
    // 存放所有的轨迹
    std::vector<std::vector<State>> all_traj;
    // 存放最优轨迹；
    std::vector<State> best_traj;
    dwa_test.getTrajectory(all_traj, best_traj);
    pose_now.x = best_traj.back().x;
    pose_now.y = best_traj.back().y;
    pose_now.angle = best_traj.back().yaw;
    odom_now.v = best_traj.back().linear_vel;
    odom_now.w = best_traj.back().angular_vel;
    cout << "odom_now.v:" << odom_now.v << ", odom_now.w:" << odom_now.w << endl;

    // 画出所有路径
    for (int i = 0; i < all_traj.size(); i++)
    {
      int k = all_traj[i].size();
      cv::line(color_map, cv::Point(all_traj[i][0].x / 0.05, all_traj[i][0].y / 0.05), cv::Point(all_traj[i][k - 1].x / 0.05, all_traj[i][k - 1].y / 0.05), cv::Scalar(0, 0, 255), 1);
    }
    // 画出最优路径
    cv::line(color_map, cv::Point(best_traj[0].x / 0.05, best_traj[0].y / 0.05), cv::Point(best_traj[best_traj.size() - 1].x / 0.05, best_traj[best_traj.size() - 1].y / 0.05), cv::Scalar(255, 255, 0), 1);
    cv::circle(color_map, cv::Point(best_traj[best_traj.size() - 1].x / 0.05, best_traj[best_traj.size() - 1].y / 0.05), 1, cv::Scalar(0, 0, 255));

    // 计算当前位置与终点的距离,到达则退出
    EVec2f pose(pose_now.x, pose_now.y);
    float dist2end = (pose - detail_path.back()).norm();
    if (dist2end < 0.15)
    {
      cout << "dist2end less 0.15" << endl;
      break;
    }
  }
  imwrite("../test_result/color_map.png", color_map);
}