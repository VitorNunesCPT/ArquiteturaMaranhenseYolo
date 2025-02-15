# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
import sys
from typing import Any
import warnings

# Suprimir avisos não críticos
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = "false"

import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference using
    Streamlit and Ultralytics YOLO models. It provides the functionalities such as loading models, configuring settings,
    uploading video files, and performing real-time inference.

    Attributes:
        st (module): Streamlit module for UI creation.
        temp_dict (dict): Temporary dictionary to store the model path.
        model_path (str): Path to the loaded model.
        model (YOLO): The YOLO model instance.
        source (str): Selected video source.
        enable_trk (str): Enable tracking option.
        conf (float): Confidence threshold.
        iou (float): IoU threshold for non-max suppression.
        vid_file_name (str): Name of the uploaded video file.
        selected_ind (list): List of selected class indices.

    Methods:
        web_ui: Sets up the Streamlit web interface with custom HTML elements.
        sidebar: Configures the Streamlit sidebar for model and inference settings.
        source_upload: Handles video file uploads through the Streamlit interface.
        configure: Configures the model and loads selected classes for inference.
        inference: Performs real-time object detection inference.

    Examples:
        >>> inf = solutions.Inference(model="path/to/model.pt")  # Model is not necessary argument.
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any):
        """Inicializa a classe Inference."""
        check_requirements("streamlit>=1.29.0")
        import streamlit as st
        
        self.st = st
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Configura a interface web Streamlit."""
        # Configuração da página deve ser a primeira chamada Streamlit
        self.st.set_page_config(
            page_title="Detector Arquitetura MA", 
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': None
            }
        )
        
        # Esconder elementos da interface que podem causar warnings
        hide_elements = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
        """
        self.st.markdown(hide_elements, unsafe_allow_html=True)

        # Título personalizado
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Detector de Arquitetura Maranhense</h1></div>"""

        # Subtítulo personalizado
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-bottom:50px;">Detecção de elementos arquitetônicos maranhenses em tempo real</h4></div>"""
        
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configura a barra lateral do Streamlit."""
        with self.st.sidebar:
            self.st.title("Configurações")
            
            # Seção de seleção da fonte de vídeo
            self.source = self.st.selectbox(
                "Fonte do Vídeo",
                ("webcam", "video"),
            )
            
            # Opção de tracking
            self.enable_trk = self.st.radio("Habilitar Tracking", ("Sim", "Não"))
            
            # Ajuste de parâmetros
            self.conf = float(
                self.st.slider("Limiar de Confiança", 0.0, 1.0, self.conf, 0.01)
            )
            self.iou = float(
                self.st.slider("Limiar IoU", 0.0, 1.0, self.iou, 0.01)
            )

        # Criar colunas para os frames
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.markdown("### Vídeo Original")
            self.org_frame = self.st.empty()
        with col2:
            self.st.markdown("### Detecções")
            self.ann_frame = self.st.empty()

    def source_upload(self):
        """Gerencia o upload de vídeo e configuração da webcam."""
        self.vid_file_name = None  # Inicializa como None
        
        if self.source == "video":
            # Upload de vídeo
            vid_file = self.st.sidebar.file_uploader(
                "Carregar Vídeo", 
                type=["mp4", "mov", "avi", "mkv"]
            )
            if vid_file is not None:
                # Salvar o vídeo temporariamente
                temp_file = "temp_video.mp4"
                with open(temp_file, "wb") as f:
                    f.write(vid_file.read())
                self.vid_file_name = temp_file
                self.st.sidebar.success(f"Vídeo carregado: {vid_file.name}")
        
        elif self.source == "webcam":
            self.vid_file_name = 0  # 0 é o índice da webcam padrão
            self.st.sidebar.info("Webcam selecionada")

    def configure(self):
        """Configura o modelo e carrega as classes selecionadas para inferência."""
        def find_model_path():
            """Procura o arquivo de peso do modelo nos caminhos possíveis."""
            possible_paths = [
                "train3/weights/best.pt",
                "runs/detect/train3/weights/best.pt",
                "runs/train3/weights/best.pt",
                "./train3/weights/best.pt",
                "../train3/weights/best.pt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
                
            raise FileNotFoundError(
                "Arquivo de peso não encontrado! Procurei em: \n" + 
                "\n".join(possible_paths)
            )

        with self.st.spinner("Carregando modelo..."):
            try:
                model_path = find_model_path()
                self.st.info(f"Modelo encontrado em: {model_path}")
                self.model = YOLO(model_path)
                class_names = list(self.model.names.values())
                self.st.success("Modelo carregado com sucesso!")
            except Exception as e:
                self.st.error(f"Erro ao carregar modelo: {str(e)}")
                self.st.error("Por favor, verifique se o arquivo best.pt existe em train3/weights/")
                sys.exit(1)

        # Multiselect com os nomes das suas classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names)
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Realiza inferência em tempo real."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        start_button = self.st.sidebar.button("Iniciar")
        
        if start_button:
            try:
                # Tentar abrir a fonte de vídeo
                if self.vid_file_name is None:
                    self.st.error("Por favor, selecione uma fonte de vídeo (webcam ou arquivo)")
                    return
                    
                cap = cv2.VideoCapture(self.vid_file_name)
                
                if not cap.isOpened():
                    self.st.error(
                        "Erro ao abrir fonte de vídeo. " +
                        "Se for webcam, verifique se está conectada. " +
                        "Se for arquivo, verifique se é um formato válido."
                    )
                    return

                # Criar botão de parar
                stop_button = self.st.button("Parar")
                self.st.info("Processando... Clique em 'Parar' para encerrar.")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        self.st.warning("Não foi possível ler o próximo frame.")
                        break

                    # Fazer predições
                    if self.enable_trk == "Sim":
                        results = self.model.track(
                            frame, 
                            conf=self.conf, 
                            iou=self.iou,
                            classes=self.selected_ind,
                            persist=True
                        )
                    else:
                        results = self.model(
                            frame,
                            conf=self.conf,
                            iou=self.iou,
                            classes=self.selected_ind
                        )

                    # Plotar resultados
                    annotated_frame = results[0].plot()

                    # Mostrar frames
                    self.org_frame.image(frame, channels="BGR", caption="Vídeo Original")
                    self.ann_frame.image(annotated_frame, channels="BGR", caption="Detecções")

                    if stop_button:
                        break

                # Limpar recursos
                cap.release()
                self.st.success("Detecção finalizada!")
                
            except Exception as e:
                self.st.error(f"Erro durante a execução: {str(e)}")
                if 'cap' in locals():
                    cap.release()
            
            finally:
                # Remover arquivo temporário se existir
                if self.source == "video" and self.vid_file_name:
                    try:
                        os.remove(self.vid_file_name)
                    except:
                        pass


if __name__ == "__main__":
    # Simplificar a execução
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None
    Inference(model=model).inference()
