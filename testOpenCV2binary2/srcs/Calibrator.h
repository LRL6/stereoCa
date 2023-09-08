#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ChessBoard.h"
#include "config.h"
#include "utils.h"
using namespace cv;
using namespace std;

/*　立体标定　*/
class Calibrator {
public:
    Size imgSize;       // 图像大小 1280x720
    Size patSize;       // 棋盘格 cols x rows
    double squareSize;  // 方格尺寸 5.f
    Mat M_L, M_R;       // 内参矩阵(左/右)
    Mat D_L, D_R;       // 畸变系数(左/右)
    vector<Mat> rvecs_L, rvecs_R;  // 旋转矩阵(左/右)
    vector<Mat> tvecs_L, tvecs_R;  // 平移矩阵(左/右)
    double error_L = 0., error_R = 0.; // 标定误差(单目)

    Mat R, T;          // 旋转矩阵（R）+矩阵（T）
    Mat E, F;          // 本质矩阵（）
    double error_LR = 0.;  // 标定误差（双目）

    bool is_calib = false; // 是否完成标定

private:
    // 存储图像
    Mat img, imgL, imgR, imgGrayL, imgGrayR, imgShow;
    vector<Mat> imgsL, imgsR;
    // 标定图片
    vector<String> calib_files, calib_files_good;
    // 世界坐标
    vector<Point3f> corners_xyz;
    vector<vector<Point2f>> corners_uv_all_L, corners_uv_all_R;
    // 像素坐标
    vector<Point2f> corners_uv_L, corners_uv_R;
    vector<vector<Point3f>> corners_xyz_all;

public:
    bool calibStereo(const string &folder,
                     int W, int H,
                     ChessBoard &chessBoard,
                     bool show = true){
        // 读取标定图片
        glob(folder, calib_files);
        for (auto &file : calib_files){
            imread2(file, img);
            d2(img, imgL, imgR);
            imgsL.push_back(imgL); imgsR.push_back(imgR);
        }

        imgSize = Size(W, H);
        patSize = Size(chessBoard.cols, chessBoard.rows);
        squareSize = chessBoard.square_size; assert(squareSize > 0);

        // 棋盘格角点世界坐标
        for (int y = 0; y < patSize.height; y++){
            for (int x = 0; x < patSize.width; x++){
                corners_xyz.emplace_back(Point3f(float(x * squareSize), float(y * squareSize), 0));
            }
        }

        cout << "start detect corner:" << endl;
        for (int i = 0; i < calib_files.size(); i++) {
            cout << "load img:" << calib_files[i];
            imgL = imgsL[i]; imgR = imgsR[i];
            bool find = detectCorners(show);
            if (find){
                calib_files_good.emplace_back(calib_files[i]);
                corners_xyz_all.emplace_back(corners_xyz);
                corners_uv_all_L.emplace_back(corners_uv_L);
                corners_uv_all_R.emplace_back(corners_uv_R);
            }
            cout << "\t detect \t" << find << endl;
        }
        cout << "corner detect sucessful !" << endl;

        cout << "start single camera calibrate: " << endl;
        // 左、右相机分别标定
        error_L = calibrateCamera(corners_xyz_all,  // 世界坐标
                                  corners_uv_all_L, // 像素坐标
                                  imgSize,            // 标定图像大小
                                  M_L,             // 相机内参
                                  D_L,                // 畸变系数
                                  rvecs_L,                // 旋转矩阵
                                  tvecs_L,                // 平移矩阵
                                  CALIB_FIX_K3);             // 鱼眼相机(k1,k2,k3)
        error_R = calibrateCamera(corners_xyz_all,
                                  corners_uv_all_R,
                                  imgSize,
                                  M_R,
                                  D_R,
                                  rvecs_R,
                                  tvecs_R,
                                  CALIB_FIX_K3);
        printInfo();
        cout << "单目标定完成!" << endl;

        cout << "开始立体标定..." << endl;
        // 双目标定
        error_LR = stereoCalibrate(corners_xyz_all,      // 标定板世界坐标
                                   corners_uv_all_L,    // 像素坐标(左)
                                   corners_uv_all_R,    // 像素坐标(右)
                                   M_L,D_L,  // 相机内参\畸变系数(左)
                                   M_R, D_R, // 相机内参\畸变系数(右)
                                   imgSize,                // 标定图像大小
                                   R,    // 旋转矩阵(右->左)
                                   T,    // 平移矩阵(右->左)
                                   E,    // 本质矩阵(同一点在左\右像素坐标系下的相互转换矩阵,单位mm)
                                   F,    // 基础矩阵
                                   CALIB_FIX_INTRINSIC,
                                   TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-5));
        printStereoInfo();
        is_calib = true;
        cout << "立体标定完成!" << endl;
        return is_calib;
    }

    bool save2xml(const string &intrinsic_file, const string &extrinsic_file) const{
        cout << "写入内参到文件:" << intrinsic_file << endl;
        FileStorage fw1(intrinsic_file, FileStorage::WRITE);
        if (fw1.isOpened()){
            fw1 << "M_L" << M_L << "D_L" << D_L <<
                "M_R" << M_R << "D_R" << D_R;
            fw1.release();
        } else{
            cerr << "无法打开内参文件:" << intrinsic_file << endl; assert(0);
        }
        cout << "写入外参到文件:" << extrinsic_file << endl;
        FileStorage fw2(extrinsic_file, FileStorage::WRITE);
        if (fw2.isOpened()){
            fw2 << "R" << R  << "T" << T;
            fw2.release();
        }
        else{
            cerr << "无法打开外参文件:" << extrinsic_file << endl; assert(0);
        }
        return true;
    }

private:
    /*　检测当前imgL\imgR中角点　*/
    bool detectCorners(bool show = true) {
        // 转换为灰度图像
        //cvtColor(imgL, imgGrayL, CV_BGR2GRAY);
        cvtColor(imgL, imgGrayL, cv::COLOR_BGR2GRAY);
        cvtColor(imgR, imgGrayR, cv::COLOR_BGR2GRAY);
        // 角点粗检测
        bool findL = findChessboardCorners(imgGrayL, patSize, corners_uv_L,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
        bool findR = findChessboardCorners(imgGrayR, patSize, corners_uv_R,
                                          CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
        if (findL && findR) {
            // 角点精检测
            cornerSubPix(imgGrayL, corners_uv_L, Size(11, 11), Size(-1, -1),
                         TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.01));
            cornerSubPix(imgGrayR, corners_uv_R, Size(11, 11), Size(-1, -1),
                         TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.01));
        }
        if (show) {
            // 显示检测信息
            drawChessboardCorners(imgL, patSize, corners_uv_L, findL);
            drawChessboardCorners(imgR, patSize, corners_uv_R, findR);
            hconcat(imgL, imgR, imgShow);
            imshow("chessBoard", imgShow);
            waitKey(1000);
            destroyWindow("chessBoard");
        }
        return (findL && findR);
    }

    /* 打印单目标定信息 */
    void printInfo() const{
        cout << "标定误差(L):" << error_L << endl;
        cout << "标定误差(R):" << error_R << endl;
        cout << "\n内参(左)\n" << M_L << endl;
        cout << "\n内参(右)\n" << M_R << endl;
        cout << "\n畸变(左)\n" << D_L << endl;
        cout << "\n畸变(右)\n" << D_R << endl;
    }

    /* 打印双目标定信息 */
    void printStereoInfo() const{
        cout << "标定误差(双目):" << error_LR << endl;
        cout << "\n旋转矩阵R\n" << R << endl;
        cout << "\n平移矩阵T\n" << T << endl;
        cout << "\n本质矩阵E(相机坐标系，mm)\n" << E << endl;
        cout << "\n基础矩阵F(像素坐标系，pixel)\n" << F << endl;
    }
};


#endif //CALIBRATOR_H
