import os
import json
import onnx
import torch
import argparse
import numpy as np
import onnxruntime
from model_segformer import segformer
import onnx_tensorrt.backend as backend


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def load_torch_model(cfg_path, pth_path, dataset):
    print("Loading model!")
    with open(cfg_path) as f:
        exp_cfg = json.load(f)
    model_config = exp_cfg["MODEL_CONFIG"]
    model = segformer(model_config, dataset, pretrained=True)
    save_dict = torch.load(pth_path)
    model.load_state_dict(save_dict["net"])
    model = model.to("cuda:0")
    model.eval()
    print("Model loaded!")
    return model


def torch2onnx(model, onnx_name, input):
    print("Converting pytorch model to onnx model!")
    input = input.to("cuda:0")
    # Export the model
    torch.onnx.export(
        model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        onnx_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["inputs"],  # the model's input names
        output_names=["output"],
    )  # the model's output names
    print("Converted pytorch model to onnx model!")


def test_onnx_model(onnx_name, pytorch_model, input):
    print("Testing onnx model!")
    ort_session = onnxruntime.InferenceSession(onnx_name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs_encoder = ort_session.run(None, ort_inputs)
    model_output = pytorch_model(input.to("cuda:0"))
    np.testing.assert_allclose(
        to_numpy(model_output[0]), ort_outs_encoder[0], rtol=1e-03, atol=1e-05
    )
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def onnx2trt(onnx_name, engine_path, dataset, batch_size):
    print("Converting onnx model to trt engine")
    onnx_model = onnx.load(onnx_name)
    engine = backend.prepare(onnx_model, device="CUDA:0",  max_workspace_size = (1 << 30))

    with open(engine_path, "wb") as f:
        f.write(engine.engine.engine.serialize())

    print("Converted onnx model to trt engine!")

    print("Testing trt engine")
    if dataset == "CULane":
        input_data = np.random.random(size=(batch_size, 3, 288, 800)).astype(np.float32)
    else:
        input_data = np.random.random(size=(batch_size, 3, 288, 512)).astype(np.float32)
    trt_output_data = engine.run(input_data)[0]

    ort_session = onnxruntime.InferenceSession(onnx_name)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs_encoder = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        trt_output_data, ort_outs_encoder[0], rtol=1e-03, atol=1e-05
    )
    print("Exported engine has been tested and the result looks good!")

def parse_args():
    parser = argparse.ArgumentParser("torch 2 onnx and onnx2trt must be seperated for CULane model")
    parser.add_argument("--exp_dir", type=str, help = "directory to the experiment's file path (e.g. ./experiments/exp0)")
    parser.add_argument("--saving_path", type=str, help = "directory to save the trt engine")
    parser.add_argument("--dataset", type=str, help = "dataset name (Tusimple or CULane)")
    parser.add_argument("--batch_size", type=int, help = "batch size of the tensorrt engine")
    parser.add_argument("--toonnx", action="store_true")
    parser.add_argument("--totrt", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    assert dataset in ["Tusimple","CULane"]
    exp_dir = args.exp_dir
    saving_path = args.saving_path
    batch_size = args.batch_size
    toonnx = args.toonnx
    totrt = args.totrt
    cfg_path = os.path.join(exp_dir, "cfg.json")
    pth_path = os.path.join(exp_dir, exp_dir.split("/")[-1] + "_best.pth")
    engine_path = os.path.join(saving_path, exp_dir.split("/")[-1] + "_batch{}.trt".format(str(batch_size)))
    onnx_name = os.path.join(saving_path, "test.onnx")
    if dataset == "CULane":
        input = torch.randn(batch_size, 3, 288, 800, requires_grad=True)
    else:
        input = torch.randn(batch_size, 3, 288, 512, requires_grad=True)

    if toonnx:
        pytorch_model = load_torch_model(cfg_path, pth_path, dataset)
        torch2onnx(pytorch_model, onnx_name, input)
        test_onnx_model(onnx_name, pytorch_model, input)
    if totrt:
        onnx2trt(onnx_name, engine_path, dataset, batch_size)
