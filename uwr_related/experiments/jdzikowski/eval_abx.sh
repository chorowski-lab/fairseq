dataset=$1
embeddings=/pio/scratch/1/i273233/w2v2_results/embeddings
zd=/pio/data/zerospeech2021
results=/pio/scratch/1/i273233/w2v2_results/results

source /pio/scratch/2/i273233/miniconda3/etc/profile.d/conda.sh
conda activate 202010-fairseq

python scripts/embeddings_abx.py $zd/wav2vec_small.pt $zd/$dataset $embeddings

conda activate zerospeech2021

for i in `basename -a $(ls -d $embeddings/*/)`
do
    cp $embeddings/cosine.yaml $embeddings/$i/meta.yaml
	zerospeech2021-evaluate -j 12 -o $results/cosine/$i --no-lexical --no-syntactic --no-semantic $zd/dataset_subset $embeddings/$i
    cp $embeddings/euclidean.yaml $embeddings/$i/meta.yaml
    zerospeech2021-evaluate -j 12 -o $results/euclidean/$i --no-lexical --no-syntactic --no-semantic $zd/dataset_subset $embeddings/$i
done

for i in `basename -a $(ls -d $embeddings/*/)`
do 
    echo $i
    echo "cosine"
    cat $results/cosine/$i/score_phonetic.csv
    echo
    echo "euclidean"
    cat $results/euclidean/$i/score_phonetic.csv
    echo
done > results_w2v2.txt