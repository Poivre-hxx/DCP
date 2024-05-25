# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog


class Ui_Dialog_dcp(object):

    def setupUi(self, Dialog):
        Dialog.setObjectName('Dialog')
        Dialog.resize(389, 426)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName('verticalLayout')
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName('horizontalLayout_7')
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setObjectName('pushButton_3')
        self.horizontalLayout_7.addWidget(self.pushButton_3)
        self.textBrowser_3 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_3.setObjectName('textBrowser_3')
        self.horizontalLayout_7.addWidget(self.textBrowser_3)
        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 7)
        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setObjectName('pushButton')
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setObjectName('textBrowser')
        self.horizontalLayout_5.addWidget(self.textBrowser)
        self.horizontalLayout_5.setStretch(0, 3)
        self.horizontalLayout_5.setStretch(1, 7)
        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName('horizontalLayout_6')
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setObjectName('pushButton_2')
        self.horizontalLayout_6.addWidget(self.pushButton_2)
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setObjectName('textBrowser_2')
        self.horizontalLayout_6.addWidget(self.textBrowser_2)
        self.horizontalLayout_6.setStretch(0, 3)
        self.horizontalLayout_6.setStretch(1, 7)
        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName('buttonBox')
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        self.pushButton.clicked.connect(Dialog.open_source)
        self.pushButton_2.clicked.connect(Dialog.open_target)
        self.pushButton_3.clicked.connect(Dialog.open_model)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def open_source(self):
        (self.fileName1, filetype) = QFileDialog.getOpenFileName(None, '请选择点云文件:', '.', 'All Files(*);;')
        if self.fileName1 != '':
            self.textBrowser.setText(self.fileName1)

    def open_target(self):
        (self.fileName2, filetype) = QFileDialog.getOpenFileName(None, '请选择点云文件:', '.', 'All Files(*);;')
        if self.fileName2 != '':
            self.textBrowser_2.setText(self.fileName2)

    def open_model(self):
        (self.fileName3, filetype) = QFileDialog.getOpenFileName(None, '请选择模型文件:', '.', 'All Files(*);;')
        if self.fileName3 != '':
            self.textBrowser_3.setText(self.fileName3)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate('Dialog', 'DCP配准'))
        self.pushButton_3.setText(_translate('Dialog', '选择模型'))
        self.pushButton.setText(_translate('Dialog', '模板路径:'))
        self.pushButton_2.setText(_translate('Dialog', '目标路径:'))
