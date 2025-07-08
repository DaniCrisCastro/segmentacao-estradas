import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from model import UNet

def predict_sample(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.eval()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    input_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0][0].cpu().numpy()

    mask_pred = (output > 0.5).astype(np.uint8) * 255

    # Aplicar colormap diretamente na máscara
    mask_color = cv2.applyColorMap(mask_pred, cv2.COLORMAP_PLASMA)

    # Mostrar com OpenCV
    cv2.imshow("Imagem Original", cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    cv2.imshow("Segmentação Colorida", mask_color)
    cv2.waitKey(3000)  # espera 3 segundos (3000 ms)
    cv2.destroyAllWindows()

    # Salvar saída
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "segmentacao_" + os.path.basename(image_path))
    cv2.imwrite(out_path, mask_color)
    print(f"✅ Segmentação salva em: {out_path}")

if __name__ == "__main__":
    image_dir = "images"
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            print(f"\n📷 Processando: {filename}")
            predict_sample(image_path)

    print("\n✅ Processamento finalizado!")
