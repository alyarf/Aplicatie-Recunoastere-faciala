import sys
import random
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QLabel,
    QPushButton, QFileDialog, QComboBox, QGroupBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from Algorithms import Algorithms

class Interface(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.algorithm = None
        self.directory = None
        self.image_path = None
        self.selected_i = None
        self.selected_j = None
        self.selected_norm = 2  
        self.selected_algorithm = None
        self.selected_k = 3
        self.selected_niv = 20
        #self.trunc_levels = []

    def init_ui(self):
        self.setWindowTitle("Face Recognition GUI")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: Arial;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #999999;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QRadioButton {
                spacing: 5px;
            }
            QComboBox {
                border: 1px solid #999999;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 6em;
            }
        """)

        # Main layout with background pattern
        main_layout = QHBoxLayout()
        
        # Left side - controls
        controls_layout = QVBoxLayout()
        
        # Database selection
        db_group = QGroupBox("DATA BASE")
        db_layout = QHBoxLayout()
        self.orl_radio = QRadioButton("ORL")
        self.orl_radio.setChecked(True)
        self.essex_radio = QRadioButton("Essex")
        self.ctoyf_radio = QRadioButton("CTOYF")
        db_layout.addWidget(self.orl_radio)
        db_layout.addWidget(self.essex_radio)
        db_layout.addWidget(self.ctoyf_radio)
        db_group.setLayout(db_layout)
        controls_layout.addWidget(db_group)

        # Database configuration
        config_group = QGroupBox("DATA BASE CONFIGURATION")
        config_layout = QVBoxLayout()
        self.config_60_40 = QRadioButton("60% training 40% testing")
        self.config_80_20 = QRadioButton("80% training 20% testing")
        self.config_90_10 = QRadioButton("90% training 10% testing")
        self.config_60_40.setChecked(True)
        config_layout.addWidget(self.config_60_40)
        config_layout.addWidget(self.config_80_20)
        config_layout.addWidget(self.config_90_10)
        config_group.setLayout(config_layout)
        controls_layout.addWidget(config_group)

        # Algorithm selection
        algo_group = QGroupBox("ALGORITHM")
        algo_layout = QVBoxLayout()
        
        self.nn_radio = QRadioButton("NN")
        
        knn_layout = QHBoxLayout()
        self.knn_radio = QRadioButton("kNN, k=")
        self.knn_combo = QComboBox()
        self.knn_combo.addItems(["3", "5", "7", "9"])
        knn_layout.addWidget(self.knn_radio)
        knn_layout.addWidget(self.knn_combo)
        
        eigenfaces_layout = QHBoxLayout()
        self.eigenfaces_radio = QRadioButton("Eigenfaces, k=")
        self.eigenfaces_combo = QComboBox()
        self.eigenfaces_combo.addItems(["20", "40", "80", "100"])
        eigenfaces_layout.addWidget(self.eigenfaces_radio)
        eigenfaces_layout.addWidget(self.eigenfaces_combo)
        
        rc_layout = QHBoxLayout()
        self.rc_radio = QRadioButton("Eigenfaces cu RC, k=")
        self.rc_combo = QComboBox()
        self.rc_combo.addItems(["3", "5", "7", "9"])
        rc_layout.addWidget(self.rc_radio)
        rc_layout.addWidget(self.rc_combo)
        
        lanczos_layout = QHBoxLayout()
        self.lanczos_radio = QRadioButton("Lanczos, k=")
        self.lanczos_combo = QComboBox()
        self.lanczos_combo.addItems(["20", "40", "80", "100"])
        lanczos_layout.addWidget(self.lanczos_radio)
        lanczos_layout.addWidget(self.lanczos_combo)

        algo_layout.addWidget(self.nn_radio)
        algo_layout.addLayout(knn_layout)
        algo_layout.addLayout(eigenfaces_layout)
        algo_layout.addLayout(rc_layout)
        algo_layout.addLayout(lanczos_layout)
        algo_group.setLayout(algo_layout)
        controls_layout.addWidget(algo_group)

        # Norm selection
        norm_group = QGroupBox("NORM")
        norm_layout = QVBoxLayout()
        self.manhattan_radio = QRadioButton("Manhattan")
        self.euclidean_radio = QRadioButton("Euclidiana")
        self.euclidean_radio.setChecked(True)
        self.infinite_radio = QRadioButton("Infinit")
        self.cosine_radio = QRadioButton("Cosinus")
        norm_layout.addWidget(self.manhattan_radio)
        norm_layout.addWidget(self.euclidean_radio)
        norm_layout.addWidget(self.infinite_radio)
        norm_layout.addWidget(self.cosine_radio)
        norm_group.setLayout(norm_layout)
        controls_layout.addWidget(norm_group)
        
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # File selection
        file_layout = QHBoxLayout()
        self.choose_file_button = QPushButton("Choose File")
        self.file_label = QLabel("No file chosen")
        self.file_label.setStyleSheet("color: #666666")
        
        file_layout.addWidget(self.choose_file_button)
        file_layout.addWidget(self.file_label)
        controls_layout.addLayout(file_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select")
        self.search_button = QPushButton("Search")
        self.statistics_button = QPushButton("Statistics")
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.search_button)
        button_layout.addWidget(self.statistics_button)
        controls_layout.addLayout(button_layout)

        # Right side - images
        images_layout = QVBoxLayout()
        
        # Image display areas
        self.original_image_label = QLabel()
        self.result_image_label = QLabel()
        
        self.original_image_label.setFixedSize(300, 300)
        self.result_image_label.setFixedSize(300, 300)
        self.original_image_label.setStyleSheet("""
            border: 1px solid #cccccc;
            background-color: white;
            border-radius: 5px;
        """)
        self.result_image_label.setStyleSheet("""
            border: 1px solid #cccccc;
            background-color: white;
            border-radius: 5px;
        """)
        
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        images_layout.addWidget(self.original_image_label)
        images_layout.addWidget(self.result_image_label)
        images_layout.addStretch()

        # Add layouts to main layout
        main_layout.addLayout(controls_layout, stretch=60)
        main_layout.addLayout(images_layout, stretch=40)
        
        self.setLayout(main_layout)

        # Connect signals
        self.manhattan_radio.toggled.connect(self.update_norm)
        self.euclidean_radio.toggled.connect(self.update_norm)
        self.infinite_radio.toggled.connect(self.update_norm)
        self.cosine_radio.toggled.connect(self.update_norm)
        self.choose_file_button.clicked.connect(self.choose_file)
        self.select_button.clicked.connect(self.configure_algorithm)
        self.search_button.clicked.connect(self.search_image)
        self.statistics_button.clicked.connect(self.generate_statistics)

    def update_norm(self):
        if self.manhattan_radio.isChecked():
            self.selected_norm = 1  
        elif self.euclidean_radio.isChecked():
            self.selected_norm = 2  
        elif self.infinite_radio.isChecked():
            self.selected_norm = 'inf'  
        elif self.cosine_radio.isChecked():
            self.selected_norm = 'cos'  
    
    def choose_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.pgm)")
        if file_dialog.exec():
            image_path = file_dialog.selectedFiles()[0]
            self.image_path = image_path
            self.file_label.setText(os.path.basename(image_path))
    
            try:
                # Extragem subdirectorul (ex: "s4") si numele fisierului (ex: "9.pgm")
                subdirectory = os.path.basename(os.path.dirname(image_path))  # ex: "s4"
                file_name = os.path.basename(image_path)  # ex: "9.pgm"
    
                # Extragem numarul persoanei (i) si al imaginii (j)
                if subdirectory.startswith('s') and file_name.endswith('.pgm'):
                    i = int(subdirectory[1:]) # Extragem numarul dupa "s"
                    j = int(os.path.splitext(file_name)[0]) # Extragem numarul inainte de ".pgm"
                    print("Poza test:" , j)
                    # Determinam tipul de training/testing
                    if self.config_60_40.isChecked():
                        training_images = 6
                    elif self.config_80_20.isChecked():
                        training_images = 8
                    elif self.config_90_10.isChecked():
                        training_images = 9
                    else:
                        raise ValueError("No training/testing configuration selected.")
    
                    # Calculam imaginile de testing
                    testing_images = 10 - training_images
                    testing_order = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]  # Ordinea completa
                    testing_images_list = testing_order[training_images:training_images + testing_images]
    
                    # Verificam daca imaginea selectata este una de testare
                    if j not in testing_images_list:
                        raise ValueError(f"Selected image {j}.pgm is not a valid testing image "
                                         f"for the current configuration.")
    
                    # Salvam `i` si `j` pentru utilizare ulterioara
                    self.selected_i = i # persoana i
                    self.selected_j = j # poza j
    
                    # Afisam informatia
                    self.file_label.setText(f"Selected Person (i): {i}, Test Image (j): {j}")
                else:
                    raise ValueError("Invalid file format or directory structure.")
            except Exception as e:
                self.file_label.setText(f"Error determining i and j: {str(e)}")
                self.selected_i = None
                self.selected_j = None
    
            # Load and display the original image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                height, width = image.shape
                bytes_per_line = width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    self.original_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.original_image_label.setPixmap(scaled_pixmap)
            else:
                self.file_label.setText("Error loading image")


    def configure_algorithm(self):
        reload_data = False

        self.selected_algorithm = None
        self.selected_k = 3
        self.selected_niv = 20
        self.algorithm = None
        
        # Configurarea alg. pe baza opt. alese
        if self.orl_radio.isChecked():
            self.directory = r"C:\PatternRecogn\bd ORL"
        elif self.essex_radio.isChecked():
            self.directory = "essex_faces"
        elif self.ctoyf_radio.isChecked():
            self.directory = "ctoyf_faces"
        
        '''
        if self.directory != new_directory:
            self.directory = new_directory
            reload_data = True
        else:
            reload_data = False
        '''

        
        if not os.path.exists(self.directory):
            self.file_label.setText(f"Error: {self.directory} directory not found!")
            return

        if self.config_60_40.isChecked():
            training_split = 6
        elif self.config_80_20.isChecked():
            training_split = 8
        else:  # 90/10 split
            training_split = 9
        
        testing_split = 10 - training_split
        
        testing_order = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]  # Ordinea completa
        testing_images_list = testing_order[training_split:training_split + testing_split]
        
        if (self.algorithm and
                (self.algorithm.training_images != training_split or
                 self.algorithm.testing_images != testing_split)):
            reload_data = True

        if self.nn_radio.isChecked():
            self.selected_algorithm = "nn"
        elif self.knn_radio.isChecked():
            self.selected_algorithm = "knn"
            self.selected_k = int(self.knn_combo.currentText())
        elif self.eigenfaces_radio.isChecked():
            self.selected_algorithm = "clasic"
            self.selected_niv = int(self.eigenfaces_combo.currentText())
                                        
        elif self.rc_radio.isChecked():
            self.selected_algorithm = "rc"
            self.selected_k = int(self.rc_combo.currentText())
            self.selected_niv = 80
            
        elif self.lanczos_radio.isChecked():
            self.selected_algorithm = "lanczos"
            self.selected_niv = int(self.lanczos_combo.currentText())
        else:
            self.selected_algorithm = None
            self.selected_k = 1
            self.selected_niv = 20

        # Initializam algoritmul
        try:
            self.algorithm = Algorithms(
                self.directory,
                training_split,
                testing_split,
                norma=self.selected_norm,
                niv_trunchiere = self.selected_niv
            )
            
            #if reload_data or not self.algorithm.pozele:
            if reload_data:
                self.algorithm.load_images()
            
            self.file_label.setText("Algorithm configured successfully!")
        
        except Exception as e:
            self.file_label.setText(f"Error configuring algorithm: {str(e)}")


    def display_image(self, image, label):
        print(f"Displaying image with shape: {image.shape}, dtype: {image.dtype}")
        if image.dtype != np.uint8:
            print("Warning: Image is not uint8. Converting...")
            image = cv2.normalize(image, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
    
      
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)


    def search_image(self):
        # Calc. imaginea pe baza optiunilor alese (config. in f anterioara)
        if not self.algorithm or not self.image_path:
            self.file_label.setText("Please configure the algorithm and select an image first!")
            return
    
        if self.selected_i is None or self.selected_j is None:
            self.file_label.setText("Error: i and j are not set. Choose a valid file first!")
            return
    
        try:
            # Apelare in functie de algoritm
            if self.selected_algorithm in ["nn", "knn"]:
                A, pozele = self.algorithm.load_images()
                poza_test = self.algorithm.poza_test(pozele, self.selected_i, self.selected_j)
                
                # Apelam alg_KNN pentru NN sau kNN
                person_id, position = self.algorithm.alg_KNN(
                    data=A,
                    poza_test=poza_test,
                    norma=self.selected_norm,
                    k=self.selected_k
                )
                
                
                self.file_label.setText(f"Identified person: {person_id}, Position: {position}")
                
                reconstruction = A[:, position].reshape(112, 92).astype(np.uint8)

                # Normalizeaza valorile pentru afisare (daca este necesar)
                #reconstruction = cv2.normalize(reconstruction, None, 0, 255, cv2.NORM_MINMAX)
                self.display_image(reconstruction, self.result_image_label)
                
            elif self.selected_algorithm in ["clasic", "lanczos"]:
                # Pentru algoritmi proiectivi
               
                person_id, position = self.algorithm.test_alg_proiectivi(
                    i=self.selected_i,
                    j=self.selected_j,
                    norma=self.selected_norm,
                    niv_trunchiere = self.selected_niv,
                    algoritm=self.selected_algorithm,
                    k=1
                )
                
                A = self.algorithm.get_A()

                reconstruction = A[:, position].reshape(112, 92).astype(np.uint8)

                self.display_image(reconstruction, self.result_image_label)
                
                self.file_label.setText(f"Identified person: {person_id}")
               
            elif self.selected_algorithm == "rc":
                person_id, position = self.algorithm.test_alg_proiectivi(
                    i=self.selected_i,
                    j=self.selected_j,
                    norma=self.selected_norm,
                    niv_trunchiere = 80,
                    algoritm=self.selected_algorithm,
                    k=self.selected_k
                )

                A = self.algorithm.get_A()

                print("after", A)

                reconstruction = A[:, position].reshape(112, 92).astype(np.uint8)

                self.display_image(reconstruction, self.result_image_label)

                self.file_label.setText(f"Identified person: {person_id}")
                
            else:
                self.file_label.setText("Algorithm not supported for search.")
    
        except Exception as e:
            self.file_label.setText(f"Error processing image: {str(e)}")


    """ PARTEA DE STATISTICI """    
    def generate_statistics(self):
        if not self.algorithm:
            self.file_label.setText("Please configure the algorithm first!")
            return
    
        try:
            self.configure_algorithm()
            A, pozele = self.algorithm.load_images()
            self.file_label.setText("Starting statistics generation...")
    
            if self.nn_radio.isChecked():
                self.algorithm.norma = self.selected_norm
                self.file_label.setText("Generating statistics for NN algorithm...")
                self.algorithm.statistici(A, pozele, algoritm="nn")
    
    
            elif self.knn_radio.isChecked():
                selected_k = int(self.knn_combo.currentText())
    
                self.algorithm.norma = self.selected_norm
    
                self.file_label.setText(f"Generating statistics for kNN algorithm with k={selected_k}...")
    
                self.algorithm.statistici(A, pozele, algoritm="knn")
    
                
    
            elif self.eigenfaces_radio.isChecked():
                selected_niv = int(self.eigenfaces_combo.currentText())
    
                self.algorithm.norma = self.selected_norm
    
                self.algorithm.niv_trunchiere = selected_niv
    
                self.file_label.setText(f"Generating statistics for Eigenfaces with truncation level={selected_niv}...")
    
                self.algorithm.statistici(A, pozele, algoritm="clasic")
    
                
    
            elif self.rc_radio.isChecked():
                selected_k = int(self.rc_combo.currentText())
    
                self.algorithm.norma = self.selected_norm
    
                self.file_label.setText(f"Generating statistics for RC with k={selected_k}...")
    
                self.algorithm.statistici(A, pozele, algoritm="rc")
    
                
    
            elif self.lanczos_radio.isChecked():
                selected_niv = int(self.lanczos_combo.currentText())
    
                self.algorithm.norma = self.selected_norm
    
                self.algorithm.niv_trunchiere = selected_niv
    
                self.file_label.setText(f"Generating statistics for Lanczos with truncation level={selected_niv}...")
    
                self.algorithm.statistici(A, pozele, algoritm="lanczos")
    
            else:
    
                self.file_label.setText("Please select an algorithm first!")
    
                return
    
    
            selected_alg = ""
    
            if self.nn_radio.isChecked():
    
                selected_alg = "nn"
    
            elif self.knn_radio.isChecked():
    
                selected_alg = "knn"
    
            elif self.eigenfaces_radio.isChecked():
    
                selected_alg = "clasic"
    
            elif self.rc_radio.isChecked():
    
                selected_alg = "rc"
    
            elif self.lanczos_radio.isChecked():
    
                selected_alg = "lanczos"
    

            file_path = f"Rezultate_{selected_alg}.txt"
    
            if os.path.exists(file_path):
                if os.path.getsize(file_path) > 0:
    
                    self.file_label.setText(f"Statistics generated successfully! Check {file_path}")
    
                    print(f"Statistics file generated: {file_path}")
    
                else:
                    self.file_label.setText("Warning: Statistics file was created but is empty")
                    print("Warning: Empty statistics file")
    
            else:
                self.file_label.setText("Error: Statistics file was not created")
                print("Error: No statistics file created")
    
                
    
        except Exception as e:
            error_message = f"Error generating statistics: {str(e)}"
            self.file_label.setText(error_message)
            print(f"Exception occurred: {error_message}")
 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Interface()
    window.show()
    sys.exit(app.exec())
