python hydra-train.py hydra.run.dir=/pio/lscratch/1/jch/fairseq/try1/finetune \
    model.w2v_path=/pio/lscratch/1/jch/fairseq/try1/checkpoints/checkpoint_last.pt  \
    --config-dir uwr_related/configs --config-name scribblelens_base_finetune
