# mcdok @ SemEval-2026 Task 13

The mcdok systems submitted to the [SemEval-2026 Task 13](https://github.com/mbzuai-nlp/SemEval-2026-Task13) shared task.

## Cite
If you use the data, code, or the information in this repository, cite the following paper.

TBA

## Source Code Structure
| File | Description |
| :- | :- |
|mcdokA.py|the script for training and inference of the mcdok submitted to subtask A|
|mcdokB.py|the script for training and inference of the mcdok submitted to subtask B|
|mcdokC.py|the script for training and inference of the mcdok submitted to subtask C|

## Installation
Clone and install the [IMGTB framework](https://github.com/kinit-sk/IMGTB), activate the conda environment.
   ```
   git clone https://github.com/kinit-sk/IMGTB.git
   cd IMGTB
   conda env create -f environment.yaml
   conda activate IMGTB
   ```

## Code Usage
1. To retrain the Gemma-3-27b model for subtask A, run the following code (official data needs to be located in the "datasets" folder). Similarly, run the code for the other subtasks and models.
   ```
   MODEL="google/gemma-3-27b-pt" python mcdokA.py --train_file_path "datasets/Task_A/train.parquet" --dev_file_path "datasets/Task_A/test_sample.parquet" --test_file_path "datasets/Task_A/test.parquet" --model "$MODEL" --prediction_file_path "predictions_A/${MODEL##*\/}.csv"
   ```
2. To run just inference, append ```--test_only``` option to the above mentioned script. Alternatively, use the IMGTB framework where you specify the path to the finetuned model and specify ```TEXT_FIELD``` to be "code" in the ```dataset``` option.
