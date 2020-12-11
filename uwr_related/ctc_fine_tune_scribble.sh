# omit the --wer-args for no evaluation

python train.py --distributed-world-size 1 ../DistSup/data \
    --save-dir ../try_slctc1 \
    `# --wer-args '("/path/to/lm/4-gram.bin","/path/to/lexicon",2,-1)'` \
    --vocab-path ./fairseq/data/handwriting/tasman.alphabet.plus.space.mode5.json \
    --post-process letter --valid-subset test --pad-to-multiples-of 4 \
    --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 0 \
    --max-update 80000 --sentence-avg --task scribblelens --arch wav2vec_ctc \
    --w2v-path ../try_sl1/checkpoint_last.pt \
    --labels ltr \
    --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
    --mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
    --feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
    --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
    --decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
    --attention-dropout 0.0 --max-tokens 10000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d \
    `#--reset-optimizer`