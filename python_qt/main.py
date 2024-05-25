from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import open3d as o3d
import pyqtgraph.opengl as gl
import sys
import random
import copy

from mainwindow_rc import Ui_MainWindow
# ---------------------filter--------------------
from filter_voxel_rc import Ui_Dialog_voxel
from filter_uniform_rc import Ui_Dialog_uniform
from filter_random_rc import Ui_Dialog_random
from filter_radius_rc import Ui_Dialog_Radius
from filter_sor_rc import Ui_Dialog_sor
from dcp_net import use_dcp

# ---------------------filter--------------------


# ---------------------registration--------------
from registration_fgr_rc import Ui_Dialog_fgr
from registration_dcp_rc import Ui_Dialog_dcp
# ---------------------registration--------------
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
from pyqtgraph.opengl import GLViewWidget
from PyQt5.QtGui import QIcon


class open3d_software(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(open3d_software, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon("Icons/ic_pride.ico"))  # 设置图标

        # ------------------------------------------------------------
        self.actionread.triggered.connect(self.read_pointcloud)  # 槽函数
        self.actionsave.triggered.connect(self.save_pointcloud)
        self.actionquit.triggered.connect(self.close)

        # -------------------------滤波-----------------------------------

        self.VoxelDialog = Filter_voxel()
        self.actionvoxel.triggered.connect(self.dialog_filter_voxel)

        self.UniformDialog = Filter_uniform()
        self.actionuniform.triggered.connect(self.dialog_filter_uniform)

        self.RandomDialog = Filter_random()
        self.actionrandom.triggered.connect(self.dialog_filter_random)

        self.RadiusDialog = Filter_radius()
        self.actionradius.triggered.connect(self.dialog_filter_radius)

        self.SorDialog = Filter_sor()
        self.actionsor.triggered.connect(self.dialog_filter_sor)

        self.FgrDialog = Rrgistration_fgr()
        self.actionFGR.triggered.connect(self.dialog_registration_fgr)

        self.DcpDialog = Registration_dcp()
        self.actionDCP.triggered.connect(self.dialog_registration_dcp)

        # ------------------------------------------------------------

        self.graphicsView = GLViewWidget(self)
        self.gridLayout.addWidget(self.graphicsView)
        self.graphicsView.setWindowTitle('pyqtgraph example: GLScatterPlotItem')  # 定义窗口标题
        self.graphicsView.setBackgroundColor(143, 153, 159)

        # self.graphicsView.opts['distance'] = 10  # 初始视角高度
        # print(self.graphicsView.cameraPosition())
        # self.graphicsView.setCameraPosition()
        # g = gl.GLGridItem()#添加网格
        # self.graphicsView.addItem(g)

    # -----------------------文件-------------------------------------
    # 读取点云
    def read_pointcloud(self):
        # print("test well")
        fileName, filetype = QFileDialog.getOpenFileName(self, "请选择点云：", '.', "All Files(*);;")
        if fileName != '':
            self.pcd = o3d.io.read_point_cloud(fileName)
            pos_view = self.pcd.get_center()
            print(pos_view)

            self.textBrowser.clear()
            # 获取 Numpy 数组
            np_points = np.asarray(self.pcd.points)
            self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
            # 创建显示对象
            self.graphicsView.clear()
            plot = gl.GLScatterPlotItem()

            # 设置显示数据
            plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=True)  # 0.05表示点的大小
            # 显示点云
            self.graphicsView.addItem(plot)
            self.graphicsView.setCameraPosition(QtGui.QVector3D(pos_view[0], pos_view[1], pos_view[2]))

    def save_pointcloud(self):
        fileName, filetype = QFileDialog.getSaveFileName(self, "文件保存：", '.', "All Files(*);;")
        if fileName != '':
            o3d.io.write_point_cloud(fileName, self.pcd)

    # -----------------------文件-------------------------------------

    # -----------------------滤波-------------------------------------

    def dialog_filter_voxel(self):
        self.VoxelDialog.show()
        if (self.VoxelDialog.exec_()):
            self.open3d_function_filter_voxel(self.VoxelDialog.get_data())

    def open3d_function_filter_voxel(self, data_str):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=float(data_str))
        print(self.pcd)

        self.textBrowser.clear()
        # 获取 Numpy 数组
        np_points = np.asarray(self.pcd.points)
        self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
        # 创建显示对象
        self.graphicsView.clear()
        plot = gl.GLScatterPlotItem()

        # 设置显示数据
        plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=False)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot)

    def dialog_filter_uniform(self):
        self.UniformDialog.show()
        if (self.UniformDialog.exec_()):
            self.open3d_function_filter_uniform(self.UniformDialog.get_data())

    def open3d_function_filter_uniform(self, data_str):
        self.pcd = self.pcd.uniform_down_sample(every_k_points=int(data_str))
        print(self.pcd)

        self.textBrowser.clear()
        # 获取 Numpy 数组
        np_points = np.asarray(self.pcd.points)
        self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
        # 创建显示对象
        self.graphicsView.clear()
        plot = gl.GLScatterPlotItem()

        # 设置显示数据
        plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=False)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot)

    def dialog_filter_random(self):
        self.RandomDialog.show()
        if (self.RandomDialog.exec_()):
            self.open3d_function_filter_random(self.RandomDialog.get_data())

    def open3d_function_filter_random(self, data_str):
        self.pcd = self.pcd.random_down_sample(sampling_ratio=float(data_str))
        print(self.pcd)

        self.textBrowser.clear()
        # 获取 Numpy 数组
        np_points = np.asarray(self.pcd.points)
        self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
        # 创建显示对象
        self.graphicsView.clear()
        plot = gl.GLScatterPlotItem()

        # 设置显示数据
        plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=False)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot)

    # 半径
    def dialog_filter_radius(self):
        self.RadiusDialog.show()
        if (self.RadiusDialog.exec_()):
            self.open3d_function_filter_radius(self.RadiusDialog.get_data())

    def open3d_function_filter_radius(self, data_str):

        c1, ind = self.pcd.remove_radius_outlier(nb_points=int(data_str[1]), radius=float(data_str[0]))
        print(int(data_str[1]))
        print(float(data_str[0]))
        self.pcd = self.pcd.select_by_index(ind)
        print(self.pcd)

        self.textBrowser.clear()
        # 获取 Numpy 数组
        np_points = np.asarray(self.pcd.points)
        self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
        # 创建显示对象
        self.graphicsView.clear()
        plot = gl.GLScatterPlotItem()

        # 设置显示数据
        plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=False)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot)

    # 统计
    def dialog_filter_sor(self):
        self.SorDialog.show()
        if (self.SorDialog.exec_()):
            self.open3d_function_filter_sor(self.SorDialog.get_data())

    def open3d_function_filter_sor(self, data_str):

        c1, ind = self.pcd.remove_statistical_outlier(nb_neighbors=int(data_str[0]), std_ratio=float(data_str[1]))
        # print(int(data_str[0]))
        # print(float(data_str[1]))
        self.pcd = self.pcd.select_by_index(ind)
        print(self.pcd)

        self.textBrowser.clear()
        # 获取 Numpy 数组
        np_points = np.asarray(self.pcd.points)
        self.textBrowser.append("点云数量：" + str(int(np_points.size / 3)))
        # 创建显示对象
        self.graphicsView.clear()
        plot = gl.GLScatterPlotItem()

        # 设置显示数据
        plot.setData(pos=np_points, color=(1, 1, 1, 1), size=0.005, pxMode=False)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot)

    # -----------------------滤波-------------------------------------------------------------

    # FPFH
    def fpfh_compute(self, pcd, radius_normal, radius_mn, radius_feature, feature_mn):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=radius_mn))  # 最大50近邻点
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid
                                                                   (radius=radius_feature,
                                                                    max_nn=feature_mn))  # 最大50近邻点
        return pcd_fpfh  # 返回FPFH特征

    # FGR配准
    def execute_fast_global_registration(self, source, target, source_fpfh, target_fpfh, distance_threshold):
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    # FGR配准
    def dialog_registration_fgr(self):
        self.FgrDialog.show()
        if (self.FgrDialog.exec_()):
            self.open3d_function_registration_fgr(self.FgrDialog.get_data())

    def open3d_function_registration_fgr(self, data_str):
        self.src_pcd = o3d.io.read_point_cloud(data_str[5])
        self.tgt_pcd = o3d.io.read_point_cloud(data_str[6])
        src_fpfh = self.fpfh_compute(self.src_pcd, float(data_str[0]), int(data_str[1]), float(data_str[2]),
                                     int(data_str[3]))
        tgt_fpfh = self.fpfh_compute(self.tgt_pcd, float(data_str[0]), int(data_str[1]), float(data_str[2]),
                                     int(data_str[3]))
        rst = self.execute_fast_global_registration(self.src_pcd, self.tgt_pcd, src_fpfh, tgt_fpfh, float(data_str[4]))
        self.rst_pcd = copy.deepcopy(self.src_pcd)  # 因此调用copy.deepcoy进行复制并保护原始点云。
        self.rst_pcd.transform(rst.transformation)

        self.textBrowser.clear()
        # 获取 Numpy 数组

        np_points1 = np.asarray(self.src_pcd.points)
        np_points2 = np.asarray(self.tgt_pcd.points)
        np_points3 = np.asarray(self.rst_pcd.points)
        self.textBrowser.append("模板数量：" + str(int(np_points1.size / 3)) + "\n")
        self.textBrowser.append("目标数量：" + str(int(np_points2.size / 3)) + "\n")
        # 创建显示对象
        pos_view = self.src_pcd.get_center()
        self.graphicsView.clear()
        plot1 = gl.GLScatterPlotItem()
        plot2 = gl.GLScatterPlotItem()
        plot3 = gl.GLScatterPlotItem()
        # 设置显示数据
        plot1.setData(pos=np_points1, color=(1, 0, 0, 1), size=3, pxMode=True)  # 0.05表示点的大小
        # 设置显示数据
        plot2.setData(pos=np_points2, color=(0, 1, 0, 1), size=3, pxMode=True)  # 0.05表示点的大小
        # 设置显示数据
        plot3.setData(pos=np_points3, color=(0, 0, 1, 1), size=3, pxMode=True)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot1)
        self.graphicsView.addItem(plot2)
        self.graphicsView.addItem(plot3)

        self.graphicsView.setCameraPosition(QtGui.QVector3D(pos_view[0], pos_view[1], pos_view[2]))
        # 是否open3d自带显示
        # self.src_pcd.paint_uniform_color([1, 0, 0])  # 点云着色
        # self.tgt_pcd.paint_uniform_color([0, 1, 0])
        # self.rst_pcd.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([self.src_pcd,self.tgt_pcd, self.rst_pcd], width=600, height=600, mesh_show_back_face=False)

    # DCP配准
    def dialog_registration_dcp(self):
        self.DcpDialog.show()
        if (self.DcpDialog.exec_()):
            self.open3d_function_registration_dcp(self.DcpDialog.get_data())

    def open3d_function_registration_dcp(self, data_str):
        self.src_pcd = o3d.read_point_cloud(data_str[0])
        self.tgt_pcd = o3d.read_point_cloud(data_str[1])
        model_path = data_str[2]
        # 模型加载调用
        T = use_dcp(self.src_pcd, self.tgt_pcd, model_path)

        self.rst_pcd = copy.deepcopy(self.src_pcd)
        self.rst_pcd.transform(T)

        self.textBrowser.clear()
        # 获取 Numpy 数组

        np_points1 = np.asarray(self.src_pcd.points)
        np_points2 = np.asarray(self.tgt_pcd.points)
        np_points3 = np.asarray(self.rst_pcd.points)
        self.textBrowser.append("模板数量：" + str(int(np_points1.size / 3)) + "\n")
        self.textBrowser.append("目标数量：" + str(int(np_points2.size / 3)) + "\n")
        # 创建显示对象
        self.graphicsView.clear()
        pos_view = self.src_pcd.get_center()
        plot1 = gl.GLScatterPlotItem()
        plot2 = gl.GLScatterPlotItem()
        plot3 = gl.GLScatterPlotItem()
        # 设置显示数据
        plot1.setData(pos=np_points1, color=(1, 0, 0, 1), size=5, pxMode=True)  # 0.05表示点的大小
        # 设置显示数据
        plot2.setData(pos=np_points2, color=(0, 1, 0, 1), size=5, pxMode=True)  # 0.05表示点的大小
        # 设置显示数据
        plot3.setData(pos=np_points3, color=(1, 1, 0, 1), size=5, pxMode=True)  # 0.05表示点的大小
        # 显示点云
        self.graphicsView.addItem(plot1)
        self.graphicsView.addItem(plot2)
        self.graphicsView.addItem(plot3)

        self.graphicsView.setCameraPosition(QtGui.QVector3D(pos_view[0], pos_view[1], pos_view[2]))

        # 是否open3d自带显示
        # self.src_pcd.paint_uniform_color([1, 0, 0])  # 点云着色
        # self.tgt_pcd.paint_uniform_color([0, 1, 0])
        # self.rst_pcd.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([self.src_pcd, self.tgt_pcd, self.rst_pcd], width=600, height=600,
        #                                   mesh_show_back_face=False)


#体素
class Filter_voxel(QtWidgets.QDialog, Ui_Dialog_voxel):
    def __init__(self):
        super(Filter_voxel, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def get_data(self):
        data_str = self.lineEdit.text()
        return data_str


#均匀
class Filter_uniform(QtWidgets.QDialog, Ui_Dialog_uniform):

    def __init__(self):
        super(Filter_uniform, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        # self.accepted.connect(self.emit_slot)

    def get_data(self):
        data_str = self.lineEdit.text()
        return data_str


#随机
class Filter_random(QtWidgets.QDialog, Ui_Dialog_random):

    def __init__(self):
        super(Filter_random, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        # self.accepted.connect(self.emit_slot)

    def get_data(self):
        data_str = self.lineEdit.text()
        return data_str


#半径
class Filter_radius(QtWidgets.QDialog, Ui_Dialog_Radius):
    def __init__(self):
        super(Filter_radius, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        # self.accepted.connect(self.emit_slot)

    def get_data(self):
        data_str1 = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        return (data_str1, data_str2)


#统计
class Filter_sor(QtWidgets.QDialog, Ui_Dialog_sor):
    def __init__(self):
        super(Filter_sor, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def get_data(self):
        data_str1 = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        return (data_str1, data_str2)


# -----------------------滤波-------------------------------------

# -----------------------配准------------------------------------
class Rrgistration_fgr(QtWidgets.QDialog, Ui_Dialog_fgr):
    def __init__(self):
        super(Rrgistration_fgr, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def get_data(self):
        data_str1 = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        data_str3 = self.lineEdit_3.text()
        data_str4 = self.lineEdit_4.text()
        data_str5 = self.lineEdit_5.text()
        return (data_str1, data_str2, data_str3, data_str4, data_str5, self.fileName1, self.fileName2)


class Registration_dcp(QtWidgets.QDialog, Ui_Dialog_dcp):
    def __init__(self):
        super(Registration_dcp, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def get_data(self):
        return (self.fileName1, self.fileName2, self.fileName3)


app = QApplication(sys.argv)
a = open3d_software()
a.setWindowTitle("Open3D software")
a.show()
sys.exit(app.exec_())
