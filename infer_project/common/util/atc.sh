onnx=$1
om=$2
bs=$3
soc=$4

input_shape="images:${bs},3,640,640"
input_fp16_nodes="images"

if [[ ${soc} == Ascend310B4 ]];then
    atc --model=${onnx} \
        --framework=5 \
        --output=${om}_bs${bs} \
        --input_format=NCHW \
        --input_shape=${input_shape} \
        --log=error \
        --soc_version=${soc} \
        --input_fp16_nodes=${input_fp16_nodes} \
        --output_type=FP16
fi

if [[ ${soc} == Ascend310P? ]];then
    atc --model=${onnx} \
        --framework=5 \
        --output=${om}_bs${bs} \
        --input_format=NCHW \
        --input_shape=${input_shape} \
        --log=error \
        --soc_version=${soc} \
        --input_fp16_nodes=${input_fp16_nodes} \
        --output_type=FP16 \
        --optypelist_for_implmode="Sigmoid" \
        --op_select_implmode=high_performance \
        --fusion_switch_file=common/util/fusion.cfg
fi
