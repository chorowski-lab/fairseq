# Changes: 
# - not distributed
# - no fp16 (data loader randomly dying)
# - using 0 workers (somehow data loading dies when >0 and multiprocessing)

# python train.py --distributed-world-size 1 --update-freq 2 /pio/scratch/2/mstyp/wav2vec/data/LibriSpeech \
#   --save-dir /pio/scratch/2/mstyp/wav2vec/try1 --num-workers 0 \
#   --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
#   --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
#   --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
#   --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce \
#   --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
#   --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
#   --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
#   --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
#   --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 \
#   --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 --min-sample-size 32000 \
#   --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1400000 --max-update 400000 \
#   --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d

# python train.py --distributed-world-size 1 --update-freq 2 /pio/scratch/2/mstyp/wav2vec/data/LibriSpeech \
#   --save-dir /pio/scratch/2/mstyp/wav2vec/try1 --num-workers 0 \
#--distsup-dir '/pio/scratch/2/mstyp/wav2vec/DistSup'

# python train.py --distributed-world-size 1 --update-freq 2 /pio/scratch/1/i283340/MGR/NewSetup/data/LibriSpeech \
#   --save-dir /pio/scratch/1/i283340/MGR/sth/wav2vec/try1 --num-workers 0 \
#--distsup-dir '/pio/scratch/1/i283340/MGR/NewSetup/DistSup'

# python train.py --distributed-world-size 1 --update-freq 2 /pio/scratch/2/mstyp/wav2vec/data/LibriSpeech \
#   --save-dir /pio/scratch/2/mstyp/wav2vec/try1 --num-workers 0 \
#   --task scribblelens --criterion wav2vec --arch wav2vec2 \
#   --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
#   --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
#   --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce \
#   --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
#   --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
#   --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
#   --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
#   --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 \
#   --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 --min-sample-size 32000 \
#   --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1400000 --max-update 400000 \
#   --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --enable-padding

python train.py --distributed-world-size 1 --update-freq 2 \
  /pio/scratch/1/i283340/MGR/NewSetup/DistSup/data \
  --save-dir ../try_sl1 --num-workers 0 \
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
  --labels "a" \
  --enable-padding # crashes without that, needs to make all lines same-size


  # python train.py --distributed-world-size 1 --update-freq 2 /pio/scratch/1/i283340/MGR/NewSetup/data/LibriSpeech \
  # --save-dir /pio/scratch/1/i283340/MGR/NewSetup/try1 --num-workers 0 \
  # --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
  # --labels ltr \
  # --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
  # --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
  # --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce \
  # --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
  # --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
  # --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
  # --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
  # --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 \
  # --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 --min-sample-size 32000 \
  # --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1400000 --max-update 400000 \
  # --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d

  # --task scribblelens --criterion wav2vec --arch wav2vec2 \
  # --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
  # --conv-feature-layers '[(512, (32,10), 5)] + [(512, (1,3), 2)] * 4 + [(512,(1,2),2)] * 2' --final-dim 256 \
  # --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce \
  # --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
  # --total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
  # --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
  # --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
  # --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 \
  # --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 --min-sample-size 32000 \
  # --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1400000 --max-update 400000 \
  # --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
  # --enable-padding # crashes without that, needs to make all lines same-size

  # scp <...>/DistSup/egs/scribblelens/tasman.alphabet.plus.space.mode5.json <dir given as data dir in run args>