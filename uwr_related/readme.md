# Updated build instructions

1. Install miniconda

2. Create the conda env
    1. Install from the yaml:
        ```
        conda env create -f environment.yml
        ```
    2. If and only if the above fails, try installing the packages:
        ```
        # Install conda somewhere...
        conda create -n 202010-fairseq -c conda-forge -c pytorch -c plotly -c nvidia python=3.7 plotly click scikit-learn jupyter jupyterlab ipympl dvc plotnine seaborn dtale nbdime cython tqdm jupytext scipy numba scikit-image scikit-fuzzy ptvsd qgrid pytorch torchvision cudatoolkit=10.1 cudatoolkit-dev=10.1 nccl cffi cython dataclasses editdistance regex sacrebleu tqdm pandas py-opencv
        pip install soundfile
        ```

3. Activate then env: `conda activate 202010-fairseq`
    
4. Install APEX
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    export TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;6.2;7.0;7.5"
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./
    cd ..
    ```

5. Clone and install fairseq
    ```
    git clone https://github.com/chorowski-lab/fairseq
    cd fairseq
    pip install --editable ./
    cd ..
    ```

6. Download data. You can also `ln -s /pio/scratch/2/jch/wav2vec/data data`
    ```
    mkdir data
    cd data
    # Fetch part of librispeech
    wget http://www.openslr.org/resources/12/dev-clean.tar.gz
    tar zxvf dev-clean.tar.gz 
    # for CTC (really needed?)
    # wget http://www.openslr.org/resources/12/dev-other.tar.gz
    # tar zxvf dev-other.tar.gz 

    # Fetch scribblelens
    mkdir scribblelens
    cd scribblelens
    url=http://www.openslr.org/resources/84/scribblelens.corpus.v1.2.zip
    wget $url
    7z e scribblelens.corpus.v1.2.zip \
        -o./ scribblelens.corpus.v1/corpora/scribblelens.paths.1.4b.zip
    ```

7. check if training works:
    - Audio:

        From the top of `fairseq` repo call: `bash uwr_related/test_cmd_audio.sh` and verify that the model _starts_ training.
    
    - Scribblelens:
        
        TODO

8. CTC:
    - Audio:
        1. Dependencies (needed for CTC evaluation - https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings):

            ```
            conda install -c conda-forge fftw
            conda install -c conda-forge cmake
            conda install -c conda-forge openblas

            cd ..
            git clone git@github.com:kpu/kenlm
            cd kenlm
            mkdir -p build
            cd build
            cmake ..
            make -j 4
            export KENLM_ROOT_DIR=/path/to/kenlm

            cd ../..
            git clone git@github.com:facebookresearch/wav2letter.git
            cd wav2letter
            git checkout tlikhomanenko-patch-1
            cd bindings/python/
            pip install -e .
            ```
        
        2. Download dictionary and generate vocab for train split:

            ```
            wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P ../data/LibriSpeech
            python examples/wav2vec/libri_labels.py ../data/LibriSpeech/valid.tsv --output-dir ../data/LibriSpeech --output-name valid
            ```

        3. Get pretrained language model (optional for audio - check `--w2l-decoder` flag in CTC evaluation script):
            TODO
            ```
            cd ..
            mkdir pretrained_models
            cd pretrained_models
            wget https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.pt
            # ??? Be sure to upper-case the language model vocab after downloading it. ???
            wget https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.dict

            ```
        
        4. Fine-tune a pretrained model with CTC:

            From the top of `fairseq` repo call: `uwr_related/bash ctc_fine_tune_audio.sh`
            
        5. Evaluate a CTC model:
            From the top of `fairseq` repo call: `uwr_related/bash ctc_eval_audio.sh`

    - Scribblelens:

        ```
        python examples/wav2vec/scribble_labels.py --data-dir ../DistSup/data/ --output-dir ../DistSup/data/ --output-name test --vocab-dir ./fairseq/data/handwriting/tasman.alphabet.plus.space.mode5.json
        ```