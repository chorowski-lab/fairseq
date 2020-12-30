# kenlm bib --w2l-decoder kenlm
# raw numbers --w2l-decoder viterbi
# transformer language model --w2l-decoder fairseqlm

python examples/speech_recognition/infer.py \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw \
--task audio_pretraining \
--nbest 1 --path /path/to/model --gen-subset dev_other \
--results-path /path/to/save/results/for/sclite --w2l-decoder viterbi \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 \
--criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter