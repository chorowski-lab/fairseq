# [!] needs to be run from fairseq main folder
RUN="smartpooling4"
RUNDIR="/pio/scratch/1/i273233/runs"
mkdir -p $RUNDIR/$RUN

python train.py --distributed-world-size 1 --update-freq 2 \
  /pio/scratch/1/i283340/MGR/NewSetup/DistSup/data `#path to Scribblelens data folder` \
  --vocab-path ./fairseq/data/handwriting/tasman.alphabet.plus.space.mode5.json `#alphabet file` \
  --save-dir $RUNDIR/$RUN --num-workers 0 \
  `#--restore-file $RUNDIR/$RUN/before.pt` \
  --keep-last-epochs 3 \
  --tensorboard-logdir $RUNDIR/$RUN --log-format simple  \
  --task scribblelens --criterion wav2vec --arch wav2vec2_scribblelens \
  --valid-subset test --pad-to-multiples-of 4 `#--max-sample-size 256` \
  --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
  --conv-feature-layers '[(64, (3, 3), (1, 2), (1, 1)), (128, (5, 5), (2, 2), (2, 2)), (256, (3,3), (1, 1), (1, 1)), (256, (3,3), (1, 2), (1, 1)), (512, (3,3), (1, 1), (1, 1)), (512, (3,3), (1, 2), (1, 1)), (512, (3,2), (2, 1), (1, 0))]' \
  --final-dim 256 \
  --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
  --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
  --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
  --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
  --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 \
  --num-negatives 100 --cross-sample-negatives 0 \
  `#--max-sample-size 250000 --min-sample-size 32000` \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 10000 --max-update 400000 \
  --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
  --labels `#can be removed for no labels` \
  --enable-padding `# crashes without that, needs to make all lines same-size` \
  --smartpooling --smartpooling-factor 3.0 --smartpooling-search-perc 0.3