function run_tracker(){ 
    ## Running the Tracker over One Sequence
    ## Expects 1=ReID model weights, 2=Detection model weights,  3=sequence path,
    #  --classes 0 tracks persons, only
    echo "Eval Sequence at "$3
    python track.py --source $3 \
                    --yolo-weights $2.pt \
                    --img 1280 \
                    --strong-sort-weights $1.pt \
                    --classes 0 \
                    --save-txt \
                    --device cpu   ## Based on allocated device
}


function run_tracker_MOT16() {
    # generate tracking results for each sequence in MOT16 
    ## Expects 1=ReID model weights, 2=Detection model weights
    echo "ReID Model: "$1
    echo "Detection Model: "$2
    for i in  MOT16-02 MOT16-04 MOT16-05 MOT16-09 MOT16-10 MOT16-11 MOT16-13 
    do
        (   
            run_tracker $1 $2 /shared_scratch/public_datasets/MOTChallenge/MOT16/train/$i/img1/ 
        ) &
        # https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
        # allow to execute up to $N jobs in parallel
        if [[ $(jobs -r -p | wc -l) -ge $N ]]
        then
            # now there are $N jobs already running, so wait here for any job
            # to be finished so there is a place to start next one.
            wait -n
        fi
    done

    # no more jobs to be started but wait for pending jobs
    # (all need to be finished)
    wait
    echo "Inference on all MOT16 sequences DONE"
}

run_tracker_MOT16 $1 $2

echo "Copying data from experiment folder to MOT16"
# create folder to place tracking results for this method
mkdir -p ../TrackEval/data/trackers/mot_challenge/MOT16-train/$2+$1/data/

## copy results to folder
cp ./runs/track/$2+$1/*.txt \
   ../TrackEval/data/trackers/mot_challenge/MOT16-train/$2+$1/data/

python ../TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 \
 --TRACKERS_TO_EVAL $2+$1 --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity \
 --USE_PARALLEL False --NUM_PARALLEL_CORES 1 >> results/$2+$1.txt

