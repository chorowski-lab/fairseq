python hydra-train.py \
  hydra.run.dir=../experiments/scribble/try3_trained `# optional, if unspecified will save to outputs/date folder`\
  --config-dir uwr_related/configs --config-name scribblelens_base
