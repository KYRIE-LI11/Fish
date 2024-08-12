## 帮助信息
### === Model Options ===
###  --version      yolov5 tags [2.0/3.1/4.0/5.0/6.0/6.1/7.0], default: 7.0
###  --model        yolov5[n/s/m/l/x], default: yolov5s
###  --bs           batch size, default: 4
### === Build Options ===
###  --type         data type [fp16/int8], default: fp16
###  --calib_bs     batch size of calibration data (int8 use only), default: 4
### === Inference Options ===
###  --output_dir   output dir, default: output
### === Environment Options ===
###  --soc          soc version [Ascend310/Ascend310P?], default: Ascend310
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al version:,model:,bs:,type:,calib_bs:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;; 
        --version) version=$2; shift 2;;
        --model) model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --type) type=$2; shift 2;;
        --calib_bs) calib_bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $version ]]; then version=7.0; fi
if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $type ]]; then type=fp16; fi
if [[ -z $calib_bs ]]; then calib_bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi

if [[ ${type} == fp16 ]] ; then
    args_info="=== onnx2om args === \n version: $version \n model: $model \n bs: $bs \n type: $type \n
                output_dir: $output_dir \n soc: $soc"
    echo -e $args_info
else
    args_info="=== onnx2om args === \nversion: $version \n model: $model \n bs: $bs \n type: $type \n calib_bs: $calib_bs \n
                output_dir: $output_dir \n soc: $soc"
    echo -e $args_info
fi

if [ ! -d ${output_dir} ]; then
  mkdir ${output_dir}
fi

model_tmp=${model}

if [ ${type} == int8 ] ; then
    echo "Starting 生成量化数据"
    python3 common/quantize/generate_data.py --img_info_file=common/quantize/img_info_amct.txt --save_path=amct_data --batch_size=${calib_bs} || exit 1
    
    if [[ ${version} == 6.1 || ${version} == 7.0 ]] && [[ ${model} == yolov5[nl] ]] ; then
        echo "Starting pre_amct"
        python3 common/quantize/calibration_scale.py --input=${model}.onnx --output=${model}_cali.onnx --mode=pre_amct || exit 1

        echo "Starting onnx模型量化"
        bash common/quantize/amct.sh ${model}_cali.onnx || exit 1
        if [[ -f ${output_dir}/result_deploy_model.onnx ]];then
            mv ${output_dir}/result_deploy_model.onnx ${model}_amct.onnx
        fi
        rm -rf ${model}_cali.onnx

        echo "Starting after_amct"
        python3 common/quantize/calibration_scale.py --input=${model}_amct.onnx --output=${model}_amct.onnx --mode=after_amct || exit 1
    else
        echo "Starting onnx模型量化"
        bash common/quantize/amct.sh ${model}.onnx || exit 1
        if [[ -f ${output_dir}/result_deploy_model.onnx ]];then
            mv ${output_dir}/result_deploy_model.onnx ${model}_amct.onnx
        fi
    fi

    model_tmp=${model}_amct
    if [[ -f ${output_dir}/result_* ]];then
        rm -rf  ${output_dir}/result_result_fake_quant_model.onnx
        rm -rf  ${output_dir}/result_quant.json
    fi
fi

echo "Starting onnx导出om模型（无后处理）"
bash common/util/atc.sh ${model_tmp}.onnx ${output_dir}/${model_tmp} ${bs} ${soc} || exit 1

echo -e "onnx导出om模型 Success \n"
