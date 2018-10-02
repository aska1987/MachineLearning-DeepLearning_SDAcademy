# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calculate.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_calculator(object):
    firstNum=0
    operator=""
    secondNum=0
    str_firstNum=""
    str_secondNum=""
    operTF=False
    def setupUi(self, calculator):
        calculator.setObjectName("calculator")
        calculator.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(calculator)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.btn_cal = QtWidgets.QPushButton(calculator)
        self.btn_cal.setGeometry(QtCore.QRect(300, 180, 75, 23))
        self.btn_cal.setObjectName("btn_cal")
        self.btn_1 = QtWidgets.QPushButton(calculator)
        self.btn_1.setGeometry(QtCore.QRect(20, 20, 75, 23))
        self.btn_1.setObjectName("btn_1")
        self.btn_2 = QtWidgets.QPushButton(calculator)
        self.btn_2.setGeometry(QtCore.QRect(110, 20, 75, 23))
        self.btn_2.setObjectName("btn_2")
        self.btn_3 = QtWidgets.QPushButton(calculator)
        self.btn_3.setGeometry(QtCore.QRect(200, 20, 75, 23))
        self.btn_3.setObjectName("btn_3")
        self.btn_4 = QtWidgets.QPushButton(calculator)
        self.btn_4.setGeometry(QtCore.QRect(20, 60, 75, 23))
        self.btn_4.setObjectName("btn_4")
        self.btn_5 = QtWidgets.QPushButton(calculator)
        self.btn_5.setGeometry(QtCore.QRect(110, 60, 75, 23))
        self.btn_5.setObjectName("btn_5")
        self.btn_6 = QtWidgets.QPushButton(calculator)
        self.btn_6.setGeometry(QtCore.QRect(200, 60, 75, 23))
        self.btn_6.setObjectName("btn_6")
        self.btn_7 = QtWidgets.QPushButton(calculator)
        self.btn_7.setGeometry(QtCore.QRect(20, 100, 75, 23))
        self.btn_7.setObjectName("btn_7")
        self.btn_8 = QtWidgets.QPushButton(calculator)
        self.btn_8.setGeometry(QtCore.QRect(110, 100, 75, 23))
        self.btn_8.setObjectName("btn_8")
        self.btn_9 = QtWidgets.QPushButton(calculator)
        self.btn_9.setGeometry(QtCore.QRect(200, 100, 75, 23))
        self.btn_9.setObjectName("btn_9")
        self.btn_0 = QtWidgets.QPushButton(calculator)
        self.btn_0.setGeometry(QtCore.QRect(200, 140, 75, 23))
        self.btn_0.setObjectName("btn_0")
        self.btn_add = QtWidgets.QPushButton(calculator)
        self.btn_add.setGeometry(QtCore.QRect(300, 20, 75, 23))
        self.btn_add.setObjectName("btn_add")
        self.btn_mius = QtWidgets.QPushButton(calculator)
        self.btn_mius.setGeometry(QtCore.QRect(300, 60, 75, 23))
        self.btn_mius.setObjectName("btn_mius")
        self.btn_multi = QtWidgets.QPushButton(calculator)
        self.btn_multi.setGeometry(QtCore.QRect(300, 100, 75, 23))
        self.btn_multi.setObjectName("btn_multi")
        self.btn_divide = QtWidgets.QPushButton(calculator)
        self.btn_divide.setGeometry(QtCore.QRect(300, 140, 75, 23))
        self.btn_divide.setObjectName("btn_divide")
        self.result = QtWidgets.QLabel(calculator)
        self.result.setGeometry(QtCore.QRect(30, 190, 101, 81))
        self.result.setText("")
        self.result.setObjectName("result")

        self.retranslateUi(calculator)
        self.buttonBox.accepted.connect(calculator.accept)
        self.buttonBox.rejected.connect(calculator.reject)
        QtCore.QMetaObject.connectSlotsByName(calculator)
     
        self.result.setText(self.str_firstNum)
        self.btn_1.clicked.connect(self.btn1_clicked)
        self.btn_2.clicked.connect(self.btn2_clicked)
        self.btn_3.clicked.connect(self.btn3_clicked)
        self.btn_4.clicked.connect(self.btn4_clicked)
        self.btn_5.clicked.connect(self.btn5_clicked)
        self.btn_6.clicked.connect(self.btn6_clicked)
        self.btn_7.clicked.connect(self.btn7_clicked)
        self.btn_8.clicked.connect(self.btn8_clicked)
        self.btn_9.clicked.connect(self.btn9_clicked)
        self.btn_0.clicked.connect(self.btn0_clicked)
        self.btn_add.clicked.connect(self.btn_add_clicked)
        self.btn_mius.clicked.connect(self.btn_minus_clicked)
        self.btn_multi.clicked.connect(self.btn_multi_clicked)
        self.btn_divide.clicked.connect(self.btn_divide_clicked)
        self.btn_cal.clicked.connect(self.btn_cal_clicked)
###마우스 클릭 이벤트
##연산자 버튼
    def btn_add_clicked(self):
        if self.str_firstNum=='' and self.operTF==False:
            pass
        else:
            self.operator='+'
            self.operTF=True
            self.firstNum=int(self.str_firstNum) #첫번째 숫자 저장'
            self.str_firstNum+=self.operator
            self.result.setText(self.str_firstNum)
    def btn_minus_clicked(self):
        if self.str_firstNum=='' and self.operTF==False:
            pass
        else:
            self.operator='-'
            self.operTF=True
            self.firstNum=int(self.str_firstNum) #첫번째 숫자 저장
            self.str_firstNum+=self.operator
            self.result.setText(self.str_firstNum)            
    def btn_multi_clicked(self):
        if self.str_firstNum=='' and self.operTF==False:
            pass
        else:
            self.operator='*'
            self.operTF=True
            self.firstNum=int(self.str_firstNum) #첫번째 숫자 저장
            self.str_firstNum+=self.operator
            self.result.setText(self.str_firstNum)
    def btn_divide_clicked(self):
        if self.str_firstNum=='' and self.operTF==False:
            pass
        else:
            self.operator='/'
            self.operTF=True
            self.firstNum=int(self.str_firstNum) #첫번째 숫자 저장
            self.str_firstNum+=self.operator
            self.result.setText(self.str_firstNum)
##숫자 버튼
    def btn1_clicked(self):
        #QMessageBox.about(self,'1','clicked')
         #아무 입력 안햇을경우
        if (self.str_firstNum)=='':
            self.str_firstNum='1'
            self.result.setText(self.str_firstNum)
        #(아무거나 입력하고), 연산자를 사용 안한 경우
        elif (self.operTF==False):  
            self.str_firstNum+='1'   
            self.result.setText(self.str_firstNum)
        #(연산자를 입력하고), 두번째 숫자에 아무것도 입력 안했을 경우
        elif (self.str_secondNum)=='':
            self.str_secondNum='1'
            self.result.setText(self.str_firstNum,self+self.str_secondNum)
        #두번째 숫자 입력중인 경우
        else:
            self.str_secondNum+='1'
            self.result.setText(self.str_firstNum,self+self.str_secondNum)
        #self.str_firstNum=str(self.firstNum)
        #self.result.display(self.firstNum.value())
        
        
    def btn2_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='2'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='2'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='2'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='2'
            self.result.setText(self.str_firstNum+self.str_secondNum)
       
           
    def btn3_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='3'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='3'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='3'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='3'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn4_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='4'
            self.result.setText(self.str_firstNum)
            
        elif (self.operTF==False):
            self.str_firstNum+='4'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='4'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='4'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn5_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='5'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='5'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='5'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='5'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn6_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='6'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='6'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='6'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='6'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn7_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='7'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='7'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='7'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='7'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        
    def btn8_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='8'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='8'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='8'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='8'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn9_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='9'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='9'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='9'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='9'
            self.result.setText(self.str_firstNum+self.str_secondNum)
    def btn0_clicked(self):
        if (self.str_firstNum)=='':
            self.str_firstNum='0'
            self.result.setText(self.str_firstNum)
        elif (self.operTF==False):
            self.str_firstNum+='0'
            self.result.setText(self.str_firstNum)
        elif (self.str_secondNum)=='':
            self.str_secondNum='0'
            self.result.setText(self.str_firstNum+self.str_secondNum)
        else:
            self.str_secondNum+='0'
            self.result.setText(self.str_firstNum+self.str_secondNum)
##완료 버튼
    def btn_cal_clicked(self):
        if self.str_secondNum!='':
            self.secondNum=int(self.str_secondNum)
            if self.operator=='+':
                self.result.setText(str(self.firstNum+self.secondNum))
            elif self.operator=='-':
                self.result.setText(str(self.firstNum-self.secondNum))
            elif self.operator=='*':
                self.result.setText(str(self.firstNum*self.secondNum))
            else:
                self.result.setText(str(self.firstNum/self.secondNum))
            self.firstNum=0
            self.operator=""
            self.secondNum=0
            self.str_firstNum=""
            self.str_secondNum=""
            self.operTF=False
#키보드 클릭 이벤트
#    def keyPressEvent(self,e):
#        if e.key()==Qt.key_1:
#            btn1_clicked(self)
    
    def retranslateUi(self, calculator):
        _translate = QtCore.QCoreApplication.translate
        calculator.setWindowTitle(_translate("calculator", "Dialog"))
        self.btn_cal.setText(_translate("calculator", "calculate"))
        self.btn_1.setText(_translate("calculator", "1"))
        self.btn_2.setText(_translate("calculator", "2"))
        self.btn_3.setText(_translate("calculator", "3"))
        self.btn_4.setText(_translate("calculator", "4"))
        self.btn_5.setText(_translate("calculator", "5"))
        self.btn_6.setText(_translate("calculator", "6"))
        self.btn_7.setText(_translate("calculator", "7"))
        self.btn_8.setText(_translate("calculator", "8"))
        self.btn_9.setText(_translate("calculator", "9"))
        self.btn_0.setText(_translate("calculator", "0"))
        self.btn_add.setText(_translate("calculator", "+"))
        self.btn_mius.setText(_translate("calculator", "-"))
        self.btn_multi.setText(_translate("calculator", "*"))
        self.btn_divide.setText(_translate("calculator", "/"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    calculator = QtWidgets.QDialog()
    ui = Ui_calculator()
    ui.setupUi(calculator)
    calculator.show()
    sys.exit(app.exec_())

