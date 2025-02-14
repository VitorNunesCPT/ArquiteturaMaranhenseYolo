from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# URL fornecido pelo aplicativo IP Webcam
camera_url = "http://192.168.100.153:8080/video"  # Substitua pelo IP exibido no aplicativo

# Abre o feed de vídeo da câmera do celular
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Erro: Não foi possível abrir o feed. Verifique o URL ou a conexão.")
    exit()

# Carrega o modelo YOLO
model = YOLO("runs/detect/train/weights/best.pt")  # Substitua pelo caminho do modelo treinado

# Configurações de rastreamento
track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Loop principal
while True:
    # Captura frame do feed da câmera
    ret, img = cap.read()
    if not ret:
        print("Erro ao capturar o frame. Verifique a conexão.")
        break

    # Aplica o modelo YOLO no frame
    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Processa os resultados
    for result in results:
        # Visualiza as detecções no frame
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                # Verifica se há caixas detectadas
                if result.boxes is not None and result.boxes.xywh is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id
                    if track_ids is not None:
                        track_ids = track_ids.int().cpu().tolist()
                        
                        # Adiciona as linhas de rastreamento
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # Centro da caixa
                            if len(track) > 30:  # Limita o histórico de rastreamento
                                track.pop(0)

                            # Desenha as linhas de rastreamento
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except Exception as e:
                print(f"Erro no rastreamento: {e}")

    # Exibe o frame processado
    cv2.imshow("Detecção na Câmera do Celular", img)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
print("Desligando...")
