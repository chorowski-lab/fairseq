########## CHANGE THIS ##################
ZEROSPEECH_EVAL_ENV=zerospeech2021 # Where the zerospeech2021-evaluate is installed
CPC_ENV=cpc
FAIRSEQ_ENV=202010-fairseq
CONDA_PATH=/pio/scratch/2/i273233/miniconda3
FAIRSEQ_PATH=/pio/scratch/1/i273233/fairseq
CPC_PATH=/pio/scratch/1/i273233/cpc
#########################################

DATASET_PATH=$1
CHECKPOINT_PATH=$2
OUTPUT_DIR=$3
MODEL_KIND=$4 # Either "wav2vec2" or "cpc"

case $MODEL_KIND in
    wav2vec2|cpc)
        ;;
    *)
        echo "Invalid MODEL_KIND! Yours: $MODEL_KIND. Valid: wav2vec2 or cpc"
        exit 1
    ;;
esac

results=$OUTPUT_DIR/results
embeddings=$OUTPUT_DIR/embeddings
mkdir -p embeddings

source $CONDA_PATH/etc/profile.d/conda.sh
SAVED_ENV=$(conda info | sed -n 's/\( \)*active environment : //p')
echo SAVED_ENV: $SAVED_ENV

ENV_TO_ACTIVATE=$CPC_ENV
if [[ $MODEL_KIND == "wav2vec2" ]]; then
    ENV_TO_ACTIVATE=$FAIRSEQ_ENV
fi
conda activate $ENV_TO_ACTIVATE

echo "$FAIRSEQ_PATH/uwr_related/embeddings_abx.py"
python $FAIRSEQ_PATH/uwr_related/embeddings_abx.py $CHECKPOINT_PATH $DATASET_PATH $embeddings --model $MODEL_KIND --path_cpc $CPC_PATH


conda activate $ZEROSPEECH_EVAL_ENV

frame_shift="0.01"
if [[ $MODEL_KIND == "wav2vec2" ]]; then
    frame_shift="0.02"
fi
echo "Frame shift is ${frame_shift}s"

metrics=(cosine euclidean)
for metric in ${metrics[@]}
do
    cat > $embeddings/$metric.yaml << EOF
author: LSTM Baseline
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  LSTM. See https://zerospeech.com/2021 for more details.
open_source: true
train_set: librispeech 100 and 960
gpu_budget: 60
parameters:
  phonetic:
    metric: ${metric}
    frame_shift: ${frame_shift}
EOF

    for i in `basename -a $(ls -d $embeddings/*/)`
    do
        cp $embeddings/$metric.yaml $embeddings/$i/meta.yaml
        zerospeech2021-evaluate -j 12 -o $results/$metric/$i --no-lexical --no-syntactic --no-semantic $DATASET_PATH $embeddings/$i
    done
done

for metric in ${metrics[@]}
do
    for i in `basename -a $(ls -d $embeddings/*/)`
    do 
        echo $i $metric
        cat $results/$metric/$i/score_phonetic.csv
        echo
    done
done > $OUTPUT_DIR/combined_results.txt

conda activate $SAVED_ENV