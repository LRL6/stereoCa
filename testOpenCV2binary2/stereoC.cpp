
#include <iostream>
#include "srcs/Calibrator.h"
#include "srcs/CoreBM.h"
#include "srcs/config.h"

using namespace std;


int main() {
	PATH path; PARA para;
	ChessBoard chessBoard;
	Calibrator calibrator;
	// 01 单目、双目标定
	calibrator.calibStereo(path.calib_dir, para.W, para.H, chessBoard, true);
	calibrator.save2xml(path.intrinsic_file, path.extrinsic_file);
	// 02 BM匹配算法
	Mat disp8, dispRGB, xyz;
	vector<String> files; glob(path.test_dir, files);
	CoreBM bm;
	bm.init(path.intrinsic_file, path.extrinsic_file, para.W, para.H);
	for (auto &file : files) {
		// 畸变校正、计算视差图
		bm.match(file, disp8);
		// 查看、保存视差图
		show(disp8, "disparaty(gray)");
		imwrite(path.disp8_img, disp8);
		//save_disp(disp8, path.disp_txt);
		applyColorMap(disp8, dispRGB, COLORMAP_JET);
		show(dispRGB, "disparaty(RGB)");
		imwrite(path.dispRGB, dispRGB);
		// 重投影计算3D坐标位置
		reproject3d(disp8, xyz, bm.Q);
		// 保存点云
		save3dPoint(xyz, path.point_cloud_txt);
	}
	return 0;
}
