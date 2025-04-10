import multiprocessing
from optparse import Option
import sys
import threading
import webbrowser

from PIL import Image
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QApplication, QGridLayout, QFileDialog, QVBoxLayout
from qfluentwidgets import FluentIcon as FIF, QConfig, TitleLabel, ScrollArea
from qfluentwidgets import FluentWindow, PushButton, \
    FluentIcon, ImageLabel, ComboBox, SubtitleLabel, OptionsSettingCard

from wrapper import judge, classify, web
from time import sleep


class Widget(ScrollArea):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = TitleLabel(text, self)
        self.layout = QVBoxLayout(self)  # 使用 QVBoxLayout 替代 QGridLayout
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)  # 设置统一的间距

        # 添加标题
        self.layout.addWidget(self.label)

        # 添加打开网页按钮
        self.button = PushButton(FluentIcon.GLOBE, '打开网页', self)
        self.layout.addWidget(self.button)
        self.button.clicked.connect(self.open_web)

        # 添加文件选择按钮
        self.button = PushButton(FluentIcon.FOLDER, '打开文件', self)
        self.layout.addWidget(self.button)
        self.button.clicked.connect(self.openFileNameDialog)

        # 添加图片预览
        self.image = ImageLabel(self)
        self.image.setText('图像预览')
        self.layout.addWidget(self.image)

        # 添加模型选择
        self.model_select = SubtitleLabel('选择模型', self)
        self.layout.addWidget(self.model_select)
        self.select = ComboBox(self)
        self.select.addItems(['Organika/sdxl-detector', 'CNNDetection'])
        self.layout.addWidget(self.select)

        # 添加检测按钮
        self.button2 = PushButton(FluentIcon.PLAY, '检测', self)
        self.button2.setEnabled(False)  # 按钮未选择文件时为灰色
        self.button2.clicked.connect(self.process_image)
        self.layout.addWidget(self.button2)

        # 添加主题设置卡片
        a = QConfig()
        card = OptionsSettingCard(
            a.themeMode,
            FluentIcon.BRUSH,
            "应用主题",
            "调整你的应用外观",
            texts=["浅色", "深色", "跟随系统设置"]
        )
        self.layout.addWidget(card)

        # 添加检测结果标签
        self.result = SubtitleLabel('检测结果', self)
        self.layout.addWidget(self.result)

        # 必须给子界面设置全局唯一的对象名
        self.setObjectName(text.replace(' ', '-'))

    # 选择图片对话框
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file = QFileDialog.getOpenFileName(self, '打开文件', '/', 'Images (*.png *.bmp *.jpg)', options=options)
        print(file)
        if file[0]:
            self.image.setImage(file[0])
            # 按比例缩放到指定高度
            self.image.scaledToHeight(300)
            # 圆角
            self.image.setBorderRadius(8, 8, 8, 8)
            # 设置按钮可用
            self.button2.setEnabled(True)
            self.img_path = file[0]
        else:
            self.image.setText('未选择图片')

    def process_image(self):
        img = Image.open(self.img_path)
        if self.select.currentText() == 'Organika/sdxl-detector':
            result = classify(img)['artificial']
        else:
            result = judge(img)
        print(result)
        self.result.setText(f'人工合成概率：{result * 100:.9f}%')

    def open_web(self):
        p = multiprocessing.Process(target=web)
        p.start()
        sleep(5)
        webbrowser.open('http://127.0.0.1:7860/')


class Window(FluentWindow):
    """ 主界面 """

    def __init__(self):
        super().__init__()

        # 创建子界面，实际使用时将 Widget 换成自己的子界面
        self.homeInterface = Widget('人工智能合成图像检测', self)
        # self.homeInterface = Demo()
        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, '主页')
        self.navigationInterface.setAcrylicEnabled(True)

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon(':/qfluentwidgets/images/logo.png'))
        self.setWindowTitle('AI合成图像检测')


def run():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()


if __name__ == "__main__":
    run()
