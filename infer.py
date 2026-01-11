import cv2
import numpy as np
import tritonclient.http as httpclient
import argparse

MODEL_NAME = "hgg_lgg"

def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Match your training (ToTensor): 0..1 float
    x = img.astype(np.float32) / 255.0

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))

    # Add batch -> NCHW
    x = np.expand_dims(x, axis=0)  # (1,3,224,224)
    return x

def infer(image_path: str, orin_ip: str):
    x = preprocess(image_path)
    
    triton_url = f"{orin_ip}:8000"
    client = httpclient.InferenceServerClient(url=triton_url)

    inp = httpclient.InferInput("input", x.shape, "FP32")
    inp.set_data_from_numpy(x)

    out = httpclient.InferRequestedOutput("output")

    res = client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    y = res.as_numpy("output")[0]  # (2,)
    pred = int(np.argmax(y))
    return y, pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain cancer inference using Triton server")
    parser.add_argument("--orin_ip", type=str, required=True, help="ORIN IP address")
    parser.add_argument("--image_path", type=str, default="test.jpg", help="Path to input image")
    args = parser.parse_args()
    
    logits, pred = infer(args.image_path, args.orin_ip)
    print("logits:", logits)
    print("pred:", pred, "(0=LGG, 1=HGG)")
