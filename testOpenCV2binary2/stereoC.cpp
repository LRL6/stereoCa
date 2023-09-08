
#include <iostream>
#include "srcs/Calibrator.h"
#include "srcs/CoreBM.h"
#include "srcs/config.h"

using namespace std;


int main() {
	PATH path; PARA para;
	ChessBoard chessBoard;
	Calibrator calibrator;
	// 01 ��Ŀ��˫Ŀ�궨
	calibrator.calibStereo(path.calib_dir, para.W, para.H, chessBoard, true);
	calibrator.save2xml(path.intrinsic_file, path.extrinsic_file);
	// 02 BMƥ���㷨
	Mat disp8, dispRGB, xyz;
	vector<String> files; glob(path.test_dir, files);
	CoreBM bm;
	bm.init(path.intrinsic_file, path.extrinsic_file, para.W, para.H);
	for (auto &file : files) {
		// ����У���������Ӳ�ͼ
		bm.match(file, disp8);
		// �鿴�������Ӳ�ͼ
		show(disp8, "disparaty(gray)");
		imwrite(path.disp8_img, disp8);
		//save_disp(disp8, path.disp_txt);
		applyColorMap(disp8, dispRGB, COLORMAP_JET);
		show(dispRGB, "disparaty(RGB)");
		imwrite(path.dispRGB, dispRGB);
		// ��ͶӰ����3D����λ��
		reproject3d(disp8, xyz, bm.Q);
		// �������
		save3dPoint(xyz, path.point_cloud_txt);
	}
	return 0;
}
