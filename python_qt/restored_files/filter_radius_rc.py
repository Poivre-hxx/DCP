# Source Generated with Decompyle++
# File: filter_radius_rc.cpython-39.pyc (Python 3.9)

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_Radius(object):
    
    def setupUi(self, Dialog):
        Dialog.setObjectName('Dialog')
        Dialog.resize(400, 300)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName('gridLayout_2')
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setTitle('')
        self.groupBox.setObjectName('groupBox')
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName('gridLayout')
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
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName('label_2')
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName('lineEdit_2')
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 7)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName('buttonBox')
        self.gridLayout_2.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate('Dialog', '鍗婂緞閲囨牱'))
        self.label.setText(_translate('Dialog', '鍗婂緞璁剧疆锛?))
        self.label_2.setText(_translate('Dialog', '鏈€灏戠偣锛?))


