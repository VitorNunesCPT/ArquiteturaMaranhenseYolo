import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Nome do arquivo de log
log_file = "terminal.log"

# Encapsular a lógica principal
def main():
    # TODO: Coloque aqui o código completo do seu script
    from ultralytics import YOLO
    import cv2
    from windowcapture import WindowCapture
    from collections import defaultdict
    import numpy as np

    offset_x = 0
    offset_y = 30
    wincap = WindowCapture(size=(1366, 780), origin=(offset_x, offset_y))

    model = YOLO("runs/detect/train/weights/best.pt")

    track_history = defaultdict(lambda: [])
    seguir = True
    deixar_rastro = True

    while True:
        img = wincap.get_screenshot()

        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        for result in results:
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        cv2.imshow("Tela", img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    print("desligando")

# Salvar tudo que é impresso no terminal
with open(log_file, "w") as f:
    with redirect_stdout(f), redirect_stderr(f):
        main()

# Indicar ao usuário que o log foi salvo
print(f"Execução completa. Log salvo em '{log_file}'.")
