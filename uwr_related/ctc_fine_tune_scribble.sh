python hydra-train.py hydra.run.dir=../experiments/scribble/try3_trained/finetune_notrain \
    model.w2v_path=/pio/scratch/2/mstyp/wav2vec/experiments/scribble/try3_trained/checkpoints/checkpoint_last.pt \
    --config-dir uwr_related/configs --config-name scribblelens_base_finetune
