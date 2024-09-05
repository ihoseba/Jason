# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:23:29 2024

@author: joseangelperez
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Repetidor de Texto')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.int_input = QLineEdit(self)
        self.int_input.setPlaceholderText('Introduce un entero')
        layout.addWidget(self.int_input)

        self.str_input1 = QLineEdit(self)
        self.str_input1.setPlaceholderText('Introduce un primer string')
        layout.addWidget(self.str_input1)

        self.str_input2 = QLineEdit(self)
        self.str_input2.setPlaceholderText('Introduce un segundo string')
        layout.addWidget(self.str_input2)

        self.button = QPushButton('Mostrar', self)
        self.button.clicked.connect(self.show_text)
        layout.addWidget(self.button)

        self.result = QTextEdit(self)
        self.result.setReadOnly(True)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def show_text(self):
        try:
            count = int(self.int_input.text())
            text = self.str_input1.text()
            self.result.setText((text + '\n') * count)
        except ValueError:
            self.result.setText('Por favor, introduce un entero válido.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
