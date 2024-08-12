import os
import yaml


def yaml_load(yaml_file='data.yaml'):
    with open(yaml_file, errors='ignore') as f:
        return yaml.safe_load(f)


def main():
    config = yaml_load('config.yaml').get('detection')

    infer_batch_size = config['infer_batch_size']
    quantize = config['quantize']
    model_name = config['weights']
    if quantize:
        model = f'output/{model_name}_amct_bs{infer_batch_size}.om'
    else:
        model = f'output/{model_name}_bs{infer_batch_size}.om'

    cmd_list = [
        'python onnx2om.py',
        f'python om_infer.py --model={model} --eval --visible'
    ]
    for cmd in cmd_list:
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    main()
