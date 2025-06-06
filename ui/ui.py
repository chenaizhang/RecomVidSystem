import os
import pandas as pd
# ui.py - 界面模块
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPalette, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QLabel,
    QVBoxLayout, QWidget, QPushButton, QDialog,
    QLineEdit, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QPainter, QPalette, QBrush, QIntValidator
import logging
from usercf import load_data_from_csv, UserCF, save_similar_users
from itemcf_wals import WALSRecommender
from item_popularity import ItemPopularityCalculator
from cluster_analysis import ClusteringPipeline
import sys
# ==================== 加载闪屏 ====================
class LoadingSplash(QSplashScreen):
    """ 带进度提示的加载闪屏 """

    def __init__(self):
        # 加载并缩放图片
        pixmap = QPixmap("preview.png").scaled(
            500, 500,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().__init__(pixmap)

        # 设置窗口属性
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background: transparent;")

        # 添加进度标签
        self.progress_label = QLabel("正在启动", self)
        self.progress_label.setGeometry(QRect(0, 450, 500, 30))
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                color: white;
                font: bold 14px;
                background: rgba(0, 0, 0, 150);
                padding: 5px;
                border-radius: 8px;
            }
        """)

        # 显示并强制刷新界面
        self.show()
        QApplication.processEvents()

class BackgroundWidget(QWidget):
    """ 自定义背景部件（唯一新增类） """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_pixmap = QPixmap("background.jpg")

    def paintEvent(self, event):
        """ 自动绘制背景 """
        painter = QPainter(self)
        # 保持比例并填充整个区域
        scaled_pixmap = self.bg_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled_pixmap)
# ==================== 主窗口 ====================

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    """ 系统主界面 """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("短视频推荐系统")
        self.setFixedSize(1000, 800)
        self._init_ui()

    def _init_ui(self):
        """ 初始化界面组件 """
        # 创建带背景的中央部件
        central_widget = BackgroundWidget(self)  # 修改点1：使用自定义背景部件
        self.setCentralWidget(central_widget)

        # 主布局（保持原有布局结构）
        main_layout = QVBoxLayout(central_widget)  # 修改点2：布局附加到背景部件
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setSpacing(30)


        # 系统标题
        title_label = QLabel("短视频推荐系统")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font: bold 32px;
                padding: 20px;
                background: rgba(255, 255, 255, 180);
                border-radius: 15px;
            }
        """)
        # 功能按钮样式
        button_style = """
                    QPushButton {
                        background-color: transparent;
                        color: white;
                        border: 2px solid white;
                        border-radius: 15px;
                        padding: 20px 40px;
                        font: bold 18px;
                        min-width: 250px;
                    }
                    QPushButton:hover {
                        background-color: rgba(255, 255, 255, 0.1); 
                        background-color: #2980b9;
                        padding: 22px 42px;
                    }
                    QPushButton:pressed {
                        background-color: rgba(255, 255, 255, 0.2);
                        background-color: #1c6da8;
                    }
                """

        # 功能按钮
        self.btn_task1 = QPushButton("相似用户分析")
        self.btn_task2 = QPushButton("视频推荐")
        self.btn_task3 = QPushButton("热度预测")
        self.btn_task4 = QPushButton("用户聚类分析")
        self.btn_task5 = QPushButton("视频聚类分析")
        self.btn_quit = QPushButton("退出系统")

        for btn in [self.btn_task1, self.btn_task2, self.btn_task3, self.btn_task4, self.btn_task5, self.btn_quit]:
            btn.setStyleSheet(button_style)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # 按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.btn_task1)
        button_layout.addWidget(self.btn_task2)
        button_layout.addWidget(self.btn_task3)
        button_layout.addWidget(self.btn_task4)
        button_layout.addWidget(self.btn_task5)
        button_layout.addWidget(self.btn_quit)
        button_layout.setSpacing(30)

        # 整合布局
        main_layout.addWidget(title_label)
        main_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)

        # 连接信号
        self.btn_task1.clicked.connect(lambda: self._show_task_window(1))
        self.btn_task2.clicked.connect(lambda: self._show_task_window(2))
        self.btn_task3.clicked.connect(lambda: self._show_task_window(3))
        self.btn_task4.clicked.connect(lambda: self._show_task_window(4))
        self.btn_task5.clicked.connect(lambda: self._show_task_window(5))
        self.btn_quit.clicked.connect(self.close)

    def _show_task_window(self, task_id):
        """ 显示任务窗口 """
        if task_id == 1:
            window = Task1Window(self)
        elif task_id == 2:
            window = Task2Window(self)
        elif task_id == 3:
            window = Task3Window(self)
        elif task_id == 4:
            window = Task4Window(self)
        elif task_id == 5:
            window = Task5Window(self)
        # 任务窗口的显示模式为非模态对话框
        window.show()

    # ==================== 任务窗口 ====================


class Task3Window(QDialog):
    """ 视频点击量预测查询窗口 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("视频热度预测")
        self.setFixedSize(600, 400)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        # 输入区域
        input_layout = QHBoxLayout()
        lbl_item = QLabel("目标视频ID:")
        lbl_item.setStyleSheet("font: bold 16px; color: #2c3e50;")

        self.input_item = QLineEdit()
        self.input_item.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        input_layout.addStretch()
        input_layout.addWidget(lbl_item)
        input_layout.addWidget(self.input_item)
        input_layout.addStretch()

        # 执行按钮
        self.btn_execute = QPushButton("热度预测")
        self.btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font: bold 16px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)  # 蓝色主题
        self.btn_execute.clicked.connect(self._execute_task)

        # 结果显示区域
        self.result_label = QLabel("预测热度将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font: bold 24px; color: #3498db;")

        # 布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_execute, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        main_layout.addStretch()

    def _execute_task(self):
        """ 执行点击量查询 """
        item_id = self.input_item.text().strip()

        if not item_id.isdigit():
            self._show_error("请输入有效的数字ID")
            return

        try:
            # 读取预测结果文件
            predictions_df = pd.read_csv('F5.csv')

            # 查找对应视频ID的预测点击量
            item_prediction = predictions_df[predictions_df['itemId'] == int(item_id)]

            if item_prediction.empty:
                user_count=0
            else:
            # 获取预测点击量
                user_count = item_prediction['user_count'].values[0]

            # 更新结果显示
            self.result_label.setText(f"视频 {item_id} 预测热度为:\n{int(user_count):,}")  # 千分位格式化

            # 显示成功消息
            QMessageBox.information(self, "成功", f"视频 {item_id} 预测热度: {int(user_count):,}")

        except FileNotFoundError:
            self._show_error("未找到预测结果文件(F5.csv)")
        except Exception as e:
            self._show_error(f"发生错误: {str(e)}")

    def _show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)


class Task1Window(QDialog):
    """ 相似用户查找窗口 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("相似用户查找")
        self.setFixedSize(800, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        # 输入区域
        input_layout = QHBoxLayout()
        lbl_user = QLabel("目标用户ID:")
        lbl_user.setStyleSheet("font: bold 16px; color: #2c3e50;")

        self.input_user = QLineEdit()
        self.input_user.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        input_layout.addStretch()
        input_layout.addWidget(lbl_user)
        input_layout.addWidget(self.input_user)
        input_layout.addStretch()

        # 执行按钮
        self.btn_execute = QPushButton("查找相似用户")
        self.btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font: bold 16px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.btn_execute.clicked.connect(self._execute_task)

        # 结果显示区域
        self.result_label = QLabel("相似用户列表将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font: 14px;")

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)  # 只保留1列（原修改点1）
        self.result_table.setHorizontalHeaderLabels(["相似用户ID","昵称","性别","年龄","粉丝数","关注的用户数"])  # 修改列标题（原修改点2）
        self.result_table.setRowCount(10)  # 预设10行，对应10个相似用户

        # 布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_execute, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_table)

    def _execute_task(self):
        """ 执行相似用户查找 """
        user_id = self.input_user.text().strip()

        if not user_id.isdigit():
            self._show_error("请输入有效的数字ID")
            return

        try:
            # 读取相似用户数据文件
            similar_users_df = pd.read_csv('F3.csv')
            userfile_df=pd.read_csv("user.csv")
            # 查找对应用户ID的相似用户
            similar_users = similar_users_df[similar_users_df['userId'] == int(user_id)]

            if similar_users.empty:
                self._show_error("未找到该用户的相似用户数据")
                return

            # 解析相似用户列表（用分号分隔的字符串）
            similar_users_list = similar_users['similarUsers'].values[0].split(';')

            # 确保有10个相似用户
            if len(similar_users_list) != 10:
                self._show_error("相似用户数据格式不正确，应为10个用分号分隔的用户ID")
                return

            # 更新表格显示
            self.result_label.setText(f"用户 {user_id} 的10个最相似用户:")

            for i, similar_id in enumerate(similar_users_list):
                self.result_table.setItem(i, 0, QTableWidgetItem(similar_id))  # 直接填充用户ID（原修改点3）
                similar_user=userfile_df[userfile_df['userId'] == int(similar_id)]
                name = similar_user['name'].values[0]
                gender=similar_user['gender'].values[0]
                age = str(similar_user['age'].values[0])
                fans = str(similar_user['fans'].values[0])
                following = str(similar_user['following'].values[0])
                self.result_table.setItem(i, 1, QTableWidgetItem(name))  # 直接填充用户ID（原修改点3）
                self.result_table.setItem(i, 2, QTableWidgetItem(gender))  # 直接填充用户ID（原修改点3）
                self.result_table.setItem(i, 3, QTableWidgetItem(age))  # 直接填充用户ID（原修改点3）
                self.result_table.setItem(i, 4, QTableWidgetItem(fans))  # 直接填充用户ID（原修改点3）
                self.result_table.setItem(i, 5, QTableWidgetItem(following))  # 直接填充用户ID（原修改点3）

            # 调整列宽和行高
            # self.result_table.resizeColumnsToContents()
            # self.result_table.resizeRowsToContents()

            # 显示成功消息
            QMessageBox.information(self, "成功", f"已找到用户 {user_id} 的10个相似用户")

        except FileNotFoundError:
            self._show_error("未找到相似用户数据文件(F3.csv)")
        except Exception as e:
            self._show_error(f"发生错误: {str(e)}")

    def _show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
class Task2Window(QDialog):
    """ 视频推荐窗口 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("个性化视频推荐")
        self.setFixedSize(800, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        # 输入区域
        input_layout = QHBoxLayout()
        lbl_user = QLabel("目标用户ID:")
        lbl_user.setStyleSheet("font: bold 16px; color: #2c3e50;")

        self.input_user = QLineEdit()
        self.input_user.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        input_layout.addStretch()
        input_layout.addWidget(lbl_user)
        input_layout.addWidget(self.input_user)
        input_layout.addStretch()

        # 执行按钮
        self.btn_execute = QPushButton("获取推荐视频")
        self.btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font: bold 16px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.btn_execute.clicked.connect(self._execute_task)

        # 结果显示区域
        self.result_label = QLabel("推荐视频列表将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font: 14px;")

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(["视频ID","长度(秒)","评论数","点赞数","播放量","分享数","标题"])
        self.result_table.setRowCount(10)  # 预设10行，对应10个推荐视频

        # 布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_execute, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_table)

    def _execute_task(self):
        """ 执行视频推荐查询 """
        user_id = self.input_user.text().strip()

        if not user_id.isdigit():
            self._show_error("请输入有效的数字ID")
            return

        try:
            # 读取推荐数据文件
            recommendations_df = pd.read_csv('Filtered_Recommendations.csv')
            itemfile_df = pd.read_csv('item.csv')

            # 查找对应用户ID的推荐视频
            user_recommendations = recommendations_df[recommendations_df['userId'] == int(user_id)]

            if user_recommendations.empty:
                self._show_error("未找到该用户的推荐视频数据")
                return

            # 解析推荐视频列表（用分号分隔的字符串）
            recommended_items = user_recommendations['itemId'].values[0].split(';')

            # 确保有10个推荐视频
            if len(recommended_items) != 10:
                self._show_error("推荐视频数据格式不正确，应为10个用分号分隔的视频ID")
                return

            # 更新表格显示
            self.result_label.setText(f"用户 {user_id} 的10个推荐视频:")


            for i, recommended_item in enumerate(recommended_items):
                video = itemfile_df[itemfile_df['itemId'] == int(recommended_item)]
                self.result_table.setItem(i, 0, QTableWidgetItem(recommended_item))  # 直接填充用户ID（原修改点3）
                length = str(video['length'].values[0])
                comment = str(video['comment'].values[0])
                like = str(video['like'].values[0])
                watch = str(video['watch'].values[0])
                share = str(video['share'].values[0])
                title = str(video['name'].values[0])[:35]
                self.result_table.setItem(i, 1, QTableWidgetItem(length))
                self.result_table.setItem(i, 2, QTableWidgetItem(comment))
                self.result_table.setItem(i, 3, QTableWidgetItem(like))
                self.result_table.setItem(i, 4, QTableWidgetItem(watch))
                self.result_table.setItem(i, 5, QTableWidgetItem(share))
                self.result_table.setItem(i, 6, QTableWidgetItem(title))


            # 调整列宽和行高
            self.result_table.resizeColumnsToContents()
            self.result_table.resizeRowsToContents()

            # 显示成功消息
            QMessageBox.information(self, "成功", f"已找到用户 {user_id} 的10个推荐视频")

        except FileNotFoundError:
            self._show_error("未找到推荐数据文件(Filtered_Recommendations.csv)")
        except Exception as e:
            self._show_error(f"发生错误: {str(e)}")

    def _show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)


class Task4Window(QDialog):
    """ 用户聚类结果查询窗口 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("用户聚类结果查询")
        self.setFixedSize(800, 600)  # 增大窗口以适应表格显示

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        # 输入区域
        input_layout = QHBoxLayout()
        lbl_user = QLabel("目标用户ID:")
        lbl_user.setStyleSheet("font: bold 16px; color: #2c3e50;")

        self.input_user = QLineEdit()
        self.input_user.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        input_layout.addStretch()
        input_layout.addWidget(lbl_user)
        input_layout.addWidget(self.input_user)
        input_layout.addStretch()

        # 执行按钮
        self.btn_execute = QPushButton("查询聚类结果")
        self.btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font: bold 16px;
            }
            QPushButton:hover { background-color: #8e44ad; }
        """)  # 紫色主题
        self.btn_execute.clicked.connect(self._execute_task)

        # 结果显示区域
        self.result_label = QLabel("聚类结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font: bold 16px; color: #9b59b6;")

        # 用户表格区域
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(["用户ID", "昵称", "性别", "年龄", "粉丝数", "关注数"])
        self.result_table.setRowCount(10)  # 预设10行，对应10个同类用户

        # 布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_execute, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_table)

    def _execute_task(self):
        """ 执行聚类结果查询 """
        user_id = self.input_user.text().strip()

        if not user_id.isdigit():
            self._show_error("请输入有效的数字ID")
            return

        try:
            # 读取聚类结果文件和用户信息文件
            cluster_df = pd.read_csv('F7.csv')
            user_df = pd.read_csv('user.csv')

            # 查找对应用户ID的聚类结果
            user_cluster = cluster_df[cluster_df['userId'] == int(user_id)]

            if user_cluster.empty:
                self._show_error("未找到该用户的聚类数据")
                return

            # 获取聚类结果
            cluster_num = user_cluster['cluster'].values[0]

            # 更新结果显示
            self.result_label.setText(f"用户 {user_id} 属于聚类: {cluster_num} - 同类用户列表")

            # 从F7.csv中获取该聚类的所有用户(排除查询用户自己)
            cluster_users = cluster_df[(cluster_df['cluster'] == cluster_num) &
                                       (cluster_df['userId'] != int(user_id))]

            if cluster_users.empty:
                self._show_error(f"聚类 {cluster_num} 中没有其他用户数据")
                return

            # 随机抽取10个用户
            if len(cluster_users) >= 10:
                sample_users = cluster_users.sample(10)
            else:
                sample_users = cluster_users.sample(len(cluster_users))

            # 清空表格
            self.result_table.clearContents()

            # 填充表格数据
            for i, (_, row) in enumerate(sample_users.iterrows()):
                user_id = str(row['userId'])
                # 获取用户详细信息
                user_info = user_df[user_df['userId'] == int(user_id)].iloc[0]

                self.result_table.setItem(i, 0, QTableWidgetItem(user_id))
                self.result_table.setItem(i, 1, QTableWidgetItem(str(user_info['name'])))
                self.result_table.setItem(i, 2, QTableWidgetItem(str(user_info['gender'])))
                self.result_table.setItem(i, 3, QTableWidgetItem(str(user_info['age'])))
                self.result_table.setItem(i, 4, QTableWidgetItem(str(user_info['fans'])))
                self.result_table.setItem(i, 5, QTableWidgetItem(str(user_info['following'])))

            # 调整列宽和行高
            # self.result_table.resizeColumnsToContents()
            # self.result_table.resizeRowsToContents()

            # 显示成功消息
            QMessageBox.information(self, "成功",
                                    f"用户 {user_id} 属于第 {cluster_num} 类\n已显示{len(sample_users)}个同类用户")

        except FileNotFoundError as e:
            if 'F7.csv' in str(e):
                self._show_error("未找到聚类结果文件(F7.csv)")
            else:
                self._show_error("未找到用户数据文件(user.csv)")
        except Exception as e:
            self._show_error(f"发生错误: {str(e)}")

    def _show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)


class Task5Window(QDialog):
    """ 视频聚类结果查询窗口 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("视频聚类结果查询")
        self.setFixedSize(800, 600)  # 保持与Task4相同的窗口大小

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)

        # 输入区域
        input_layout = QHBoxLayout()
        lbl_item = QLabel("目标视频ID:")
        lbl_item.setStyleSheet("font: bold 16px; color: #2c3e50;")

        self.input_item = QLineEdit()
        self.input_item.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                min-width: 200px;
            }
        """)

        input_layout.addStretch()
        input_layout.addWidget(lbl_item)
        input_layout.addWidget(self.input_item)
        input_layout.addStretch()

        # 执行按钮
        self.btn_execute = QPushButton("查询聚类结果")
        self.btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 10px;
                padding: 12px 30px;
                font: bold 16px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """)  # 红色主题，与Task4区分
        self.btn_execute.clicked.connect(self._execute_task)

        # 结果显示区域
        self.result_label = QLabel("聚类结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font: bold 24px; color: #e74c3c;")  # 大号红色字体
        #用户表格区域
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(["视频ID","长度(秒)","评论数","点赞数","播放量","分享数","标题"])
        self.result_table.setRowCount(10)
        # 布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.btn_execute, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_table)

    def _execute_task(self):
        """ 执行聚类结果查询 """
        item_id = self.input_item.text().strip()

        if not item_id.isdigit():
            self._show_error("请输入有效的数字ID")
            return

        try:
            # 读取聚类结果文件
            cluster_df = pd.read_csv('F6.csv')
            item_df = pd.read_csv('item.csv')

            # 查找对应视频ID的聚类结果
            item_cluster = cluster_df[cluster_df['itemId'] == int(item_id)]

            if item_cluster.empty:
                self._show_error("未找到该视频的聚类数据")
                return

            # 获取聚类结果
            cluster_num = item_cluster['cluster'].values[0]

            # 更新结果显示
            self.result_label.setText(f"视频 {item_id} 属于聚类: {cluster_num}-同类视频列表")
            cluster_items = cluster_df[(cluster_df['cluster']==cluster_num)&(cluster_df['userId']!=int(item_id))]
            if cluster_items.empty:
                self._show_error(f"聚类{cluster_num}中没有其他视频数据")
                return
            #随机抽取10个视频
            if len(cluster_items) >= 10:
                sample_items = cluster_items.sample(10)
            else:
                sample_items = cluster_items.sample(len(cluster_items))
            self.result_table.clearContents()

            for i, (_, row) in enumerate(sample_items.iterrows()):
                item_id = str(row['itemId'])
                item_info = item_df[item_df['itemId'] == int(item_id)].iloc[0]
                self.result_table.setItem(i,0,QTableWidgetItem(item_id))
                self.result_table.setItem(i,1,QTableWidgetItem(str(item_info['length'])))
                self.result_table.setItem(i,2,QTableWidgetItem(str(item_info['comment'])))
                self.result_table.setItem(i,3,QTableWidgetItem(str(item_info['like'])))
                self.result_table.setItem(i,4,QTableWidgetItem(str(item_info['watch'])))
                self.result_table.setItem(i, 5, QTableWidgetItem(str(item_info['share'])))
                self.result_table.setItem(i, 6, QTableWidgetItem(str(item_info['name'][:35])))

            self.result_table.resizeColumnsToContents()
            self.result_table.resizeRowsToContents()

            # 显示成功消息
            QMessageBox.information(self, "成功", f"视频 {item_id} 属于第 {cluster_num} 类\n 已显示{len(sample_items)}个同类用户")
        except FileNotFoundError as e:
            if 'F6.csv' in str(e):
                self._show_error("未找到聚类结果文件(F6.csv)")
            else:
                self._show_error("未找到视频数据文件(item.csv)")
        except Exception as e:
            self._show_error(f"发生错误: {str(e)}")

    def _show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 显示加载界面
    splash = LoadingSplash()
      # 初始化数据
    splash.close()

    # 显示主界面
    window = MainWindow()
    window.show()

    sys.exit(app.exec())