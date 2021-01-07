# kenlm bib --w2l-decoder kenlm ### <- doesn't work yet
# raw numbers --w2l-decoder viterbi
# transformer language model --w2l-decoder fairseqlm ### <- doesn't work yet

python examples/speech_recognition/infer.py ../DistSup/data \
--vocab-path ./fairseq/data/handwriting/tasman.alphabet.plus.space.mode5.json \
--task scribblelens --nbest 1 \
--path ../try_sl1/checkpoint_best.pt \
--gen-subset test --results-path ../ctc_eval_ls1 \
--w2l-decoder viterbi --lm-model ../kenlm/build/bin \
--lm-weight 2 --word-score -1 --sil-weight 0 \
--criterion ctc --labels --max-tokens 10000 \
--post-process letter --num-workers 0