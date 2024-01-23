import os
import sys
from PyQt5 import QtWidgets

from interface_graphique.utils import TtoS


class MainView(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainView, self).__init__(*args, **kwargs)
        self.op_dest = "filtered_data/"
        self.alpha_dest = "alphabet/transparent/"
        self.editFiles = [item for item in os.listdir(self.op_dest) if ".webp" in item]
        self.file_map = {i: i.replace(".webp", "").split() for i in self.editFiles}
        # self.setStyleSheet("background-image: url(images/logo2.png);")

        self.initUI()

    def initUI(self):
        self.setWindowTitle('ChatSing')
        self.setGeometry(100, 100, 1700, 900)  # Adjust size as needed

        textToSignTab = TtoS(self)

        self.setCentralWidget(textToSignTab)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainView()
    mainWin.show()
    sys.exit(app.exec_())
