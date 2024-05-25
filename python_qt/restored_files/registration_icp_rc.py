# Source Generated with Decompyle++
# File: registration_icp_rc.cpython-39.pyc (Python 3.9)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_Dialog_icp(object):
    
    def setupUi(self, Dialog_icp):
        Dialog_icp.setObjectName('Dialog_icp')
        Dialog_icp.resize(400, 431)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog_icp)
        self.gridLayout_2.setObjectName('gridLayout_2')
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_icp)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName('buttonBox')
        self.gridLayout_2.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(Dialog_icp)
        self.groupBox.setTitle('')
        self.groupBox.setObjectName('groupBox')
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName('gridLayout')
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName('pushButton')
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setObjectName('textBrowser')
        self.horizontalLayout_5.addWidget(self.textBrowser)
        self.horizontalLayout_5.setStretch(0, 3)
        self.horizontalLayout_5.setStretch(1, 7)
        self.gridLayout.addLayout(self.horizontalLayout_5, 2, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName('horizontalLayout_6')
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName('pushButton_2')
        self.horizontalLayout_6.addWidget(self.pushButton_2)
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_2.setObjectName('textBrowser_2')
        self.horizontalLayout_6.addWidget(self.textBrowser_2)
        self.horizontalLayout_6.setStretch(0, 3)
        self.horizontalLayout_6.setStretch(1, 7)
        self.gridLayout.addLayout(self.horizontalLayout_6, 3, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName('label')
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName('lineEdit')
        self.horizontalLayout.addWidget(self.lineEdit)
        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 7)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName('horizontalLayout_7')
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName('label_7')
        self.horizontalLayout_7.addWidget(self.label_7)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_5.setObjectName('lineEdit_5')
        self.horizontalLayout_7.addWidget(self.lineEdit_5)
        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 7)
        self.gridLayout.addLayout(self.horizontalLayout_7, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 1, 0, 1, 1)
        self.retranslateUi(Dialog_icp)
        self.buttonBox.accepted.connect(Dialog_icp.accept)
        self.buttonBox.rejected.connect(Dialog_icp.reject)
        self.pushButton.clicked.connect(Dialog_icp.open_source)
        self.pushButton_2.clicked.connect(Dialog_icp.open_target)
        QtCore.QMetaObject.connectSlotsByName(Dialog_icp)

    
    def open_source(self):
        print('1')
        (self.fileName1, filetype) = QFileDialog.getOpenFileName(self, '璇烽€夋嫨鐐逛簯锛?, '.', 'All Files(*);;')
        if self.fileName1 != '':
            self.textBrowser.setText(self.fileName1)

    
    def open_target(self):
        print('2')
        (self.fileName2, filetype) = QFileDialog.getOpenFileName(self, '璇烽€夋嫨鐐逛簯锛?, '.', 'All Files(*);;')
        if self.fileName2 != '':
            self.textBrowser_2.setText(self.fileName2)

    
    def retranslateUi(self, Dialog_icp):
        _translate = QtCore.QCoreApplication.translate
        Dialog_icp.setWindowTitle(_translate('Dialog_icp', 'ICP(鐐瑰鐐?閰嶅噯'))
        self.pushButton.setText(_translate('Dialog_icp', '妯℃澘璺緞锛?))
        self.pushButton_2.setText(_translate('Dialog_icp', '鐩爣璺緞锛?))
        self.label.setText(_translate('Dialog_icp', '璺濈闃堝€硷細'))
        self.lineEdit.setText(_translate('Dialog_icp', '1'))
        self.label_7.setText(_translate('Dialog_icp', '鏈€澶ц凯浠ｆ鏁帮細'))
        self.lineEdit_5.setText(_translate('Dialog_icp', '100'))


