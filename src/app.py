from typing import Optional
import sys
import os
import pickle
import threading
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
from omegaconf import OmegaConf
import torch
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QApplication, QWidget, QTabWidget, 
    QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QSlider, QFileDialog, QProgressBar, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QPainter, QColor, QIcon
from PyQt6.QtCore import Qt, QEvent, QObject, QPoint, pyqtSignal
from model import VAE
from train import extract_features
import utils


class OptimizationParams(QWidget):
    """A PyQt widget for configuring and displaying optimization parameters"""

    def __init__(
        self, 
        alpha: float = 1e-1, 
        beta: float = 1, 
        gamma: float = 1e-4,
        delta: float = 1e-4, 
        lr: float = 0.02, 
        steps: int = 100
    ) -> None:
        """Initializes the UI and parameters

        Args:
            alpha (float): Weight for the first L1 loss term in the search loss function
            beta (float): Weight for the second L1 loss term in the search loss function
            gamma (float): Weight for the third L1 loss term in the search loss function
            delta (float): Weight for the L2 loss term in the search loss function
            lr (float): Learning rate for the optimization process
            steps (int): Number of optimization steps
        """

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lr = lr
        self.steps = steps

        self.init_ui()

    def __create_field(self, label: str, value: float | int) -> tuple[QHBoxLayout, QLineEdit]:
        """Creates a horizontal layout with label and a QLineEdit field

        Args:
            label (str): The text to display in the QLabel
            value (float | int): The initial value to set in the QLineEdit field

        Returns:
            (tuple[QHBoxLayout, QLineEdit]): A tuple containing the created layout and the QLineEdit field
        """

        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        field = QLineEdit()
        field.setText(str(value))
        layout.addWidget(field)

        return layout, field

    def init_ui(self) -> None:
        """Initializes the user interface"""

        main_layout = QVBoxLayout()

        search_loss_path = os.path.join(os.path.dirname(__file__), '../assets/search_loss.html')
        with open(search_loss_path, "r", encoding="utf-8") as f:
            search_loss = f.read()

        search_loss_label = QLabel(search_loss)
        search_loss_label.setStyleSheet("font-family: \"Times New Roman\", Times, serif;font-size: 20px")
        main_layout.addWidget(search_loss_label)

        steps_layout, self.steps_field = self.__create_field("Steps:", self.steps)
        alpha_layout, self.alpha_field = self.__create_field("Alpha:", self.alpha)
        beta_layout, self.beta_field = self.__create_field("Beta:", self.beta)
        gamma_layout, self.gamma_field = self.__create_field("Gamma:", self.gamma)
        delta_layout, self.delta_field = self.__create_field("Delta:", self.delta)
        lr_layout, self.lr_field = self.__create_field("Learning rate:", self.lr)

        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save_params)

        layouts = [steps_layout, alpha_layout, beta_layout, gamma_layout, delta_layout, lr_layout]
        for layout in layouts:
            main_layout.addLayout(layout)
        main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)
        self.setWindowTitle("Optimization Params")
        self.resize(400, 300)

    def __check_field(
        self, 
        text: str, 
        field_name: str, 
        target_type: type, 
        min_val: Optional[float] = None, 
        max_val: Optional[float] = None
    ) -> int | float:
        """Validates and converts a field value
    
        Args:
            text (str): The text input from the field
            field_name (str): Name of the field for error messages
            target_type (type): Desired type
            min_val (float): Minimum allowed value (inclusive), optional
            max_val (float): Maximum allowed value (inclusive), optional
        
        Returns:
            (int|float): The converted value
            
        Raises:
            ValueError: If the field is empty, not convertible, or out of range
        """

        if not text.strip():
            raise ValueError(f"{field_name} field cannot be empty")
        
        try:
            value = target_type(text)
        except ValueError:
            raise ValueError(f"{field_name} must be a valid {target_type.__name__}")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{field_name} must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{field_name} must be at most {max_val}")
            
        return value

    def save_params(self) -> None:
        """Saves the parameters after validation"""

        try:
            self.steps = self.__check_field(self.steps_field.text(), "Steps", int, min_val=1)
            self.alpha = self.__check_field(self.alpha_field.text(), "Alpha", float, min_val=0)
            self.beta = self.__check_field(self.beta_field.text(), "Beta", float, min_val=0)
            self.gamma = self.__check_field(self.gamma_field.text(), "Gamma", float, min_val=0)            
            self.delta = self.__check_field(self.delta_field.text(), "Delta", float, min_val=0)           
            self.lr = self.__check_field(self.lr_field.text(), "Learning rate", float, min_val=1e-10, max_val=0.1)
            
            self.close()
            
        except ValueError as e:
            # Handle invalid input (non-numeric or out of range)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", str(e))
        except Exception as e:
            # Handle any other unexpected errors
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


class SaveDialog(QDialog):
    """A QDialog for saving directions"""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Save direction")

        QBtn = QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        message = QLabel("Direction name:")
        self.name = QLineEdit()

        layout.addWidget(message)
        layout.addWidget(self.name)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class PhotorobotApp(QMainWindow):
    """A main application window for the Photorobot VAE"""
    
    image_optimization_step = pyqtSignal()
    progress_bar_step = pyqtSignal(int) 
    add_slider = pyqtSignal(torch.Tensor)

    def __init__(
        self, 
        vae: VAE, 
        grouped_directions: dict[dict[str, torch.Tensor]], 
        device: torch.device, 
        config: OmegaConf
    ) -> None:
        """Initializes the PhotorobotApp main window.
        
        Sets up the main application window, initializes signals, UI components, and default parameters for image editing

        Args:
            vae (VAE): The variational autoencoder model used for image generation
            grouped_directions (dict[dict[str, torch.Tensor]]): A dictionary of grouped latent directions for image manipulation
            device (torch.device): The PyTorch device for tensor operations
            config (OmegaConf): Configuration object
        """ 

        super().__init__()

        self.vae = vae
        self.device = device
        self.latent_dim = config['latent_dim']
        self.base_img_width = config['img_width']
        self.base_img_height = config['img_height']
        self.img_width = int(2.5 * self.base_img_width)
        self.img_height = int(2.5 * self.base_img_height)

        self.grouped_directions = grouped_directions | {'Custom': {}}
        self.grouped_directions_coeff = {g: dict.fromkeys(d, 0) for g, d in grouped_directions.items()} | {'Custom': {}}
        self.var = 1 # latent vector variance

        self.z = torch.zeros(config['latent_dim'])

        self.paint_mode = False
        self.is_drawing = False
        self.last_point = QPoint()
        self.bin_mask = None
        self.brush_size = 15

        self.image_optimization_step.connect(self.update_image)
        self.progress_bar_step.connect(self.update_progress_bar)
        self.add_slider.connect(self.add_custom_slider)

        self.params = OptimizationParams()

        self.init_ui()

    def update_progress_bar(self, value: int) -> None:
        """Updates progress bar value
        
        Args:
            value (int): Progress bar value in percentage
        """

        self.progress_bar.setValue(value)

    def init_ui(self) -> None:
        """Initializes the user interface for the Photorobot VAE application"""

        menu_bar = self.menuBar()
        
        menu_structure = {
            "File": [
                ("Open", None),
                ("Save", None),
                None,  # Separator
                ("Load using encoder", self.load_encoder),
                ("Load using optimization", self.start_load_optimization),
                None,  # Separator
                ("Import directions", None),
                ("Export directions", None),
                None,  # Separator
                ("Export sketch", self.export_sketch),
                ("Exit", lambda: os._exit(0))
            ],
            "Reset": [
                ("Reset sliders", self.reset_sliders),
                ("Reset latent", self.reset_latent)
            ],
            "Search": [
                ("Select", self.toggle_paint_mode),
                ("Params", self.set_optimization_params),
                None,  # Separator
                ("Optimize", self.start_search_direction)
            ]
        }

        for menu_name, actions in menu_structure.items():
            menu = menu_bar.addMenu(menu_name)
            for action_info in actions:
                if action_info is None:
                    menu.addSeparator()
                else:
                    name, callback = action_info
                    action = menu.addAction(name)
                    if callback:
                        action.triggered.connect(callback)

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        right_layout = QVBoxLayout()
        left_layout = QVBoxLayout()
        
        self.setCentralWidget(central_widget)

        image = self.decode_latent_to_QImage(self.z * self.var)

        aligner_layout = QHBoxLayout()
        aligner_label = QLabel("Align:")
        self.aligner_slider = QSlider(Qt.Orientation.Horizontal)
        self.aligner_slider.setRange(0, 120)
        self.aligner_slider.setValue(100)
        self.aligner_slider.valueChanged.connect(self.update_image)
        aligner_layout.addWidget(aligner_label)
        aligner_layout.addWidget(self.aligner_slider)
        left_layout.addLayout(aligner_layout)

        self.right_sliders = {}
        self.tabs = self.__init_tabs()

        right_layout.addWidget(self.tabs)

        self.image = QLabel(self)
        self.pixmap = QPixmap(image)
        self.original_size = self.pixmap.size()
        self.bin_mask = QPixmap(self.pixmap.size()) 
        self.bin_mask.fill(Qt.GlobalColor.transparent)         
        self.image.setPixmap(self.pixmap)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image)

        brush_layout = QHBoxLayout()
        self.brush_label = QLabel("Brush Size:")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)

        for widget in (self.brush_label, self.brush_slider):
            widget.setVisible(False)
            brush_layout.addWidget(widget)

        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        left_layout.addLayout(brush_layout)

        self.progress_bar = QProgressBar()  
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.button = QPushButton("Sample")
        self.button.clicked.connect(self.sample)
        left_layout.addWidget(self.button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.image.setMouseTracking(True)
        self.image.installEventFilter(self)

        self.setWindowTitle("Photorobot VAE")
        self.showMaximized() 

    def __init_tabs(self) -> QTabWidget:
        """Initializes a QTabWidget containing a tab with sliders for each group of directions

        Returns:
            QTabWidget: The fully constructed tab widget with sliders grouped by category
        """

        self.tabs = QTabWidget(self)

        for group in self.grouped_directions:
            tab = QWidget()
            tab_layout = QVBoxLayout()
            for direction in self.grouped_directions[group]:
                slider_layout = QVBoxLayout()
                label = QLabel(direction)
                slider_layout.addWidget(label)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-200, 200)
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, g=group, d=direction: self.update_direction_coeff(value, g, d))
                slider_layout.addWidget(slider)
                self.right_sliders[f'{group}_{direction}'] = slider
                tab_layout.addLayout(slider_layout)
            tab_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tab.setLayout(tab_layout)

            scroll = QScrollArea()
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab)
            
            self.tabs.addTab(scroll, self.invert_icon(f"assets/icons/{group.lower()}.png"), group)

        return self.tabs

    @torch.inference_mode
    def decode_latent_to_QImage(self, latent: torch.Tensor) -> QImage:
        """Decodes a latent tensor into a grayscale QImage using a Variational Autoencoder

        Args:
            latent (torch.Tensor): Input latent tensor

        Returns:
            (QImage): QImage of size (self.img_width, self.img_height) in Format_Grayscale8.
        """

        denorm_img = utils.denormalize(self.vae.decoder(latent).squeeze(0, 1))
        resized_img = cv2.resize(denorm_img, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_LANCZOS4)
        qimg = QImage(resized_img.data, self.img_width, self.img_height, self.img_width, QImage.Format.Format_Grayscale8)

        return qimg
    
    def update_direction_coeff(self, value: int, group: str, direction: str) -> None:
        """Updates the direction coefficient in a specified group
        
        Args:
            value (int): The coefficient value as an integer percentage
            group (str): The group name where the direction coefficient should be updated
            direction (str): The specific direction within the group to update
        """

        self.grouped_directions_coeff[group][direction] = value / 100.0
        self.update_image()

    def update_brush_size(self, value: int) -> None:
        """Updates brush size
        
        Args:
            value (int): The brush size value
        """

        self.brush_size = value

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """Handles mouse events for drawing on a label in paint mode

        Args:
            source (QObject): The object that generated the event
            event (QEvent): The event to be processed

        Returns:
            (bool): True if the event was handled, False otherwise
        """

        if source == self.image and self.paint_mode:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True
                pos = self.map_to_pixmap(event.pos())
                self.last_point = pos
                self.draw_on_mask(pos)
                return True
            elif event.type() == QEvent.Type.MouseMove and self.is_drawing:
                pos = self.map_to_pixmap(event.pos())
                self.draw_on_mask(pos)
                self.last_point = pos
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = False
                return True
    
        return super().eventFilter(source, event)

    def map_to_pixmap(self, pos: QPoint) -> QPoint:
        """Maps a position from the QLabel coordinate space to the corresponding coordinates 
        within the displayed QPixmap, taking into account center alignment

        Args:
            pos (QPoint): The position in the QLabel's coordinate space to map

        Returns:
            (QPoint): The corresponding position within the QPixmap, clamped to its bounds
        """

        pixmap_rect = self.pixmap.rect()
        image_rect = self.image.rect()
        
        # Сonsidering (AlignmentFlag.AlignCenter)
        pixmap_x = (image_rect.width() - pixmap_rect.width()) // 2
        pixmap_y = (image_rect.height() - pixmap_rect.height()) // 2
        
        mapped_x = pos.x() - pixmap_x
        mapped_y = pos.y() - pixmap_y
        
        # Bounded coords with pixmap size
        mapped_x = max(0, min(mapped_x, pixmap_rect.width() - 1))
        mapped_y = max(0, min(mapped_y, pixmap_rect.height() - 1))
        
        return QPoint(mapped_x, mapped_y)

    def draw_on_mask(self, position: QPoint) -> None:
        """Draws on the binary mask using the current brush settings

        Args:
            position (QPoint): The current position of the cursor where drawing is applied
        """
        painter = QPainter(self.bin_mask)
        pen = QPen(QColor(255, 255, 255, 255), self.brush_size, Qt.PenStyle.SolidLine, 
                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        if self.is_drawing and not self.last_point.isNull():
            painter.drawLine(self.last_point, position)
        else:
            painter.drawPoint(position)
            
        painter.end()
        self.display_image()

    def display_image(self) -> None:
        """Displays the current image with a semi-transparent binary mask overlay"""

        result_pixmap = QPixmap(self.pixmap)
        painter = QPainter(result_pixmap)
        painter.setOpacity(0.6)
        painter.drawPixmap(0, 0, self.bin_mask)
        painter.end()
        self.image.setPixmap(result_pixmap)

    def update_image(self) -> None:
        """Updates the displayed image based on the current variance value and feature directions"""

        self.var = self.aligner_slider.value() / 100.0

        features = self.__calc_features()
        z = (self.z + sum(features)) * self.var
        image = self.decode_latent_to_QImage(z)

        self.pixmap = QPixmap(image)
        self.bin_mask = QPixmap(self.pixmap.size())
        self.bin_mask.fill(Qt.GlobalColor.transparent)
        self.display_image()

    def toggle_paint_mode(self) -> None:
        """Toggles the paint mode on or off"""

        self.paint_mode = not self.paint_mode
        self.brush_slider.setVisible(self.paint_mode)
        self.brush_label.setVisible(self.paint_mode)
        if not self.paint_mode:
            self.bin_mask.fill(Qt.GlobalColor.transparent)
            self.display_image()

    def get_binary_mask(self) -> torch.Tensor:
        """Converts the stored bin_mask image to a binary mask and resizes it to match base image dimensions

        Returns:
            (torch.Tensor): A binary mask as a PyTorch tensor
        """

        mask_image = self.bin_mask.toImage()
        width = mask_image.width()
        height = mask_image.height()
        
        ptr = mask_image.bits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
        arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
        
        mask_array = (arr[:, :, 3] > 0).astype(np.uint8)
        mask_resized = cv2.resize(mask_array, dsize=(self.base_img_width, self.base_img_height), interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(mask_resized)

    def start_search_direction(self) -> None:
        """Initiates the optimization process in a separate thread and displays the progress bar"""

        self.progress_bar.setVisible(True)
        thread = threading.Thread(target=self.search_direction)
        thread.start()

    def start_load_optimization(self) -> None:
        """Initiates the process of loading and optimizing an image file"""

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Images (*.png *.jpg *.jpeg *.gif)"
        )

        if file_name:
            self.progress_bar.setVisible(True)
            thread = threading.Thread(target=self.load_optimization, args=[file_name])
            thread.start()

    def __calc_features(self) -> None:
        """Calculate features by multiplying direction values with their coefficients"""

        return [    
            self.grouped_directions[group][direct] * self.grouped_directions_coeff[group][direct]
            for group in self.grouped_directions
            for direct in self.grouped_directions[group]
            if self.grouped_directions_coeff[group][direct] != 0
        ]

    def search_direction(self) -> None:
        """Performs optimization to find a search direction for image manipulation"""

        features = self.__calc_features()
        init_z = self.z.clone().detach()
        z = init_z + torch.randn_like(init_z) * 0.05
        z.requires_grad_(True)

        with torch.no_grad():
            source_img = self.vae.decoder((init_z + sum(features)) * self.var)

        mask = self.get_binary_mask()
        optim = torch.optim.AdamW([z], lr=self.params.lr)
        loss_fn = torch.nn.L1Loss()

        alpha = self.params.alpha * mask.numel() / mask.sum().item()
        beta = self.params.beta
        gamma = self.params.gamma
        delta = self.params.delta

        for i in range(self.params.steps):
            opt_img = self.vae.decoder((z + sum(features)) * self.var)
            loss = -alpha * loss_fn(opt_img * mask, source_img * mask) \
                + beta * loss_fn(opt_img * (1-mask), source_img * (1-mask)) \
                + gamma * loss_fn(init_z, z) \
                + delta * torch.sum(z ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.z = z.clone().detach()
            self.progress_bar_step.emit(int(i / self.params.steps * 100))
            self.image_optimization_step.emit()
        
        self.z = init_z

        self.add_slider.emit(z.clone().detach() - init_z)

        self.progress_bar.setVisible(False)

    def add_custom_slider(self, direction: torch.Tensor) -> None:
        """Adds a custom slider to the UI for a given direction tensor

        Args:
            direction (torch.Tensor): The direction tensor
        """

        save_dlg = SaveDialog()

        if not save_dlg.exec():
            return
        
        direction_name = save_dlg.name.text()

        self.grouped_directions['Custom'][direction_name] = direction
        self.grouped_directions_coeff['Custom'][direction_name] = 1

        tab = self.tabs.widget(7).widget()
        tab_layout = tab.layout()
    
        slider_layout = QVBoxLayout()
        label = QLabel(direction_name)
        slider_layout.addWidget(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-200, 200)
        slider.setValue(100)
        slider.valueChanged.connect(lambda value, g='Custom', d=direction_name: self.update_direction_coeff(value, g, d))
        slider_layout.addWidget(slider)
        tab_layout.addLayout(slider_layout)
        self.right_sliders[f'Custom_{direction_name}'] = slider

        self.update_image()

    def set_optimization_params(self) -> None:
        """Shows optimization params window"""

        self.params.show()

    def load_encoder(self) -> None:
        """Opens a file dialog to select an image file, processes the image, and encodes it into VAE latent space"""

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Images (*.png *.jpg *.jpeg *.gif)"
        )
        
        # TODO: crop and resize in separate window

        img = transforms(image=np.array(Image.open(file_name).convert('RGB').resize((160, 160), Image.Resampling.LANCZOS)))['image'] # TODO: fix transforms global var
        self.z = self.vae.encoder(img.unsqueeze(0))[0]
        self.update_image()

    def load_optimization(self, file_name: str) -> None:
        """Loads and optimizes an image using a VGG-based perceptual loss

        Args:
            file_name (str): Path to the input image file
        """

        img = transforms(image=np.array(Image.open(file_name).convert('RGB').resize((160, 160), Image.Resampling.LANCZOS)))['image']
        img = img.unsqueeze(0).requires_grad_(True) 
        
        # TODO: take out opt params to OptParams
        z_opt = self.vae.encoder(img)[0].clone().detach()
        z_opt.requires_grad_(True)
        optim = torch.optim.AdamW([z_opt], lr=0.02) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        loss_fn = torch.nn.MSELoss()

        for i in range(100):
            img_opt = self.vae.decoder(z_opt)[0].requires_grad_(True)
            vgg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            input_features = extract_features(img.repeat(1, 3, 1, 1))
            reconstruction_features = extract_features(img_opt.repeat(1, 3, 1, 1))
            
            for (inp, rec) in zip(input_features, reconstruction_features):
                vgg_loss = vgg_loss + loss_fn(inp, rec)

            vgg_loss = vgg_loss + 0.001 * torch.sum(z_opt**2)

            optim.zero_grad()
            vgg_loss.backward()
            optim.step()
            scheduler.step(vgg_loss)

            self.z = z_opt.clone().detach()

            self.image_optimization_step.emit()
            self.progress_bar_step.emit(int(i))
        
        self.progress_bar.setVisible(False)

    def export_sketch(self) -> None:
        """Saves sketch as jpg file"""

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save file",
            "",
            "Images (*.jpg)"
        )

        self.image.pixmap().save(file_name)

    def sample(self) -> None:
        """Samples random latent Z from normal Gaussian distribution"""

        self.z = torch.randn(self.latent_dim)
        self.update_image()

    def reset_sliders(self) -> None:
        """Resets all sliders and direction coefficients to their default values and updates the image"""

        for group in list(self.grouped_directions):
            for direction in list(self.grouped_directions[group]):
                self.right_sliders[f'{group}_{direction}'].setValue(0)         
                self.grouped_directions_coeff[group][direction] = 0      
        self.aligner_slider.setValue(100)
        self.update_image()

    def reset_latent(self) -> None:
        """Resets the latent vector to a zero tensor"""

        self.z = torch.zeros(self.latent_dim)
        self.update_image()

    def invert_icon(self, icon_path: str) -> QIcon:
        """Inverts the colors of an icon while preserving its alpha channel

        Args:
            icon_path (str): Path to the icon image fil

        Returns:
            QIcon: A Qt icon object with inverted colors
        """

        img = Image.open(icon_path).convert("RGBA") 

        r, g, b, a = img.split()

        # Invert only color channels
        rgb_inverted = ImageOps.invert(Image.merge("RGB", (r, g, b)))

        # Join with original alpha channel
        inverted_img = Image.merge("RGBA", (*rgb_inverted.split(), a))

        img_bytes = io.BytesIO()
        inverted_img.save(img_bytes, format="PNG")  
        pixmap = QPixmap()
        pixmap.loadFromData(img_bytes.getvalue())

        return QIcon(pixmap)  


if __name__ == "__main__":
    config = utils.config

    # torch.manual_seed(42)
    device = utils.device

    _, transforms = utils.get_transforms()

    params_path = os.path.join(os.path.dirname(__file__), config['params_path'])
    directions_path = os.path.join(os.path.dirname(__file__), config['directions_path'])
    vae = VAE((config['img_height'], config['img_width']), config['img_channels'], config['latent_dim'])
    vae.load_state_dict(torch.load(params_path, weights_only=True, map_location=device))
    vae.eval()

    with open(directions_path, "rb") as f:
        group_directions = pickle.load(f)

    app = QApplication(sys.argv)
    window = PhotorobotApp(vae, group_directions, device, config)
    window.show()
    sys.exit(app.exec())
