import onnxruntime as ort
import numpy as np
import torch
import time

from torchmetrics.classification import MulticlassAccuracy

from datamodule import CIFAR10DataModule
from model import ResnetModule

def test_checkpoint_inference(model, test_loader, device):
    results = []
    labels = []

    for batch in test_loader:

        images, label = batch

        images = images.to(device)

        # Run inference on the batch of images
        with torch.no_grad():
            outputs = model(images)

        outputs = outputs.cpu().detach().numpy()

        # Process outputs as needed
        outputs = np.argmax(outputs, axis=1)

        results.append(outputs[0])
        labels.append(label[0].item())

    return results, labels

def test_onnx_inference(onnx_model, test_loader, fp16=False):
    results = []
    labels = []

    for batch in test_loader:
        images, label = batch
        images = images.numpy()

        if fp16:
            images = images.astype(np.float16)

        # Run inference on the batch of images
        input_name = onnx_model.get_inputs()[0].name

        # Run inference
        outputs = onnx_model.run(None, {input_name: images})

        # Process outputs as needed
        outputs = np.argmax(outputs[0], axis=1)

        results.append(outputs[0])
        labels.append(label[0].item())

    return results, labels

if __name__ == "__main__":
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = CIFAR10DataModule()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    model = ResnetModule.load_from_checkpoint("checkpoints/epoch=29-step=42210-v1.ckpt")
    model.eval()

    onnx_model_path = "models/resnet.onnx"
    onnx = ort.InferenceSession(onnx_model_path)

    onnx_fp16_model_path = "models/resnet_fp16.onnx"
    onnx_fp16 = ort.InferenceSession(onnx_fp16_model_path)
    metric = MulticlassAccuracy(num_classes=10)

    start = time.time()
    results, labels = test_checkpoint_inference(model, test_loader, device)
    end = time.time()

    acc = metric(torch.tensor(results), torch.tensor(labels))

    print(f"PyTorch inference time: {end - start:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")

    start = time.time()
    results, labels = test_onnx_inference(onnx, test_loader)
    end = time.time()

    acc = metric(torch.tensor(results), torch.tensor(labels))
    
    print(f"Onnx inference time: {end - start:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")

    start = time.time()
    results, labels = test_onnx_inference(onnx_fp16, test_loader, fp16=True)
    end = time.time()

    acc = metric(torch.tensor(results), torch.tensor(labels))

    print(f"Onnx FP16 inference time: {end - start:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")



