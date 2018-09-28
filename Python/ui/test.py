# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):   
        Dialog.setObjectName("Dialog")
        Dialog.resize(392, 274)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.txtboxname = QtWidgets.QLineEdit(Dialog)
        self.txtboxname.setGeometry(QtCore.QRect(180, 30, 161, 20))
        self.txtboxname.setObjectName("txtboxname")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.txtboxage = QtWidgets.QLineEdit(Dialog)
        self.txtboxage.setGeometry(QtCore.QRect(180, 70, 161, 20))
        self.txtboxage.setObjectName("txtboxage")
        self.lblDisplay = QtWidgets.QLabel(Dialog)
        self.lblDisplay.setGeometry(QtCore.QRect(20, 120, 321, 81))
        self.lblDisplay.setText("")
        self.lblDisplay.setObjectName("lblDisplay")
        self.btncalculate = QtWidgets.QPushButton(Dialog)
        self.btncalculate.setGeometry(QtCore.QRect(300, 220, 75, 23))
        self.btncalculate.setObjectName("btncalculate")
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "WAHT IS YOUR NAME?"))
        self.label_2.setText(_translate("Dialog", "HOW OLD AER YOU?"))
        self.btncalculate.setText(_translate("Dialog", "Calculate!"))


if __name__ == "__main__":  
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
