python hydra-train.py \
  hydra.run.dir=/pio/lscratch/1/jch/fairseq/try1 `# optional, if unspecified will save to outputs/date folder`\
  --config-dir uwr_related/configs --config-name scribblelens_base
