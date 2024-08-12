import os
import yaml


def yaml_load(yaml_file='data.yaml'):
    with open(yaml_file, errors='ignore') as f:
        return yaml.safe_load(f)


def main():
    config_file = 'config.yaml'
    config_yaml = yaml_load(config_file).get('detection')

    weights = config_yaml['weights']
    infer_batch_size = config_yaml['infer_batch_size']
    quantize = config_yaml['quantize']
    soc = config_yaml['soc']
    infer_type = 'int8' if quantize else 'fp16'

    cmd = f'bash common/onnx2om.sh --model={weights} --bs={infer_batch_size} '\
        f'--calib_bs={infer_batch_size} --soc={soc} --type={infer_type}'

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
