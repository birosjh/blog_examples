"""
Creates a Lightning Module wrapping Resnet50 and exports it to ONNX format.
"""

import torch
import onnx

from onnxconverter_common import float16

from model import ResnetModule

    
def export_to_onnx(model, input_shape, output_path):
    model.eval()
    dummy_input = torch.randn(input_shape).to(model.device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=20
    )

def export_fp16_onnx(model_path, output_path):
    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)


if __name__ == "__main__":

    model = ResnetModule.load_from_checkpoint("checkpoints/epoch=29-step=42210-v1.ckpt")

    input_shape = (1, 3, 32, 32)
    output_path = "models/resnet.onnx"
    fp16_output_path = "models/resnet_fp16.onnx"

    export_to_onnx(model, input_shape, output_path)
    
    export_fp16_onnx(output_path, fp16_output_path)