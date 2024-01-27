from PyQt5 import QtWidgets, QtGui, QtCore

class IW(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(IW, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("Initialization...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.spinner = QtWidgets.QLabel()
        self.spinner.setAlignment(QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie("images/waiting2.gif")
        self.spinner.setMovie(self.movie)
        self.movie.start()
        self.layout.addWidget(self.spinner)

        self.setLayout(self.layout)