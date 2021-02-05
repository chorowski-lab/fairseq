# [!] needs to be run from fairseq main folder
export RUN="zerospeech_w2v2_base_abx/0"
export RUNDIR="/pio/scratch/1/i273233/runs"
mkdir -p $RUNDIR/$RUN

python hydra-train.py \
  hydra.run.dir=$RUNDIR/$RUN `# optional, if unspecified will save to outputs/date folder`\
  --config-dir uwr_related/configs --config-name audio_w2v2_base_abx