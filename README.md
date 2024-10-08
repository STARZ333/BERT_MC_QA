
### BERT Multiple Choice & Question Answering

#### How to start?

1. Clone the project repository:
   ```bash
   git clone https://github.com/huggingface/transformers
   ```

2. Navigate into the project directory:
   ```bash
   cd transformers
   ```

3. Install the project dependencies:
   ```bash
   pip install .
   ```

4. Navigate into the example folder of your choice (e.g., `transformers/examples/pytorch/question-answering`):
   ```bash
   cd transformers/examples/pytorch/question-answering
   ```

5. Install the required dependencies for the example:
   ```bash
   pip install -r requirements.txt
   ```

#### Prepare dataset and train

1. Prepare training dataset and validation dataset
   ```bash
    python generate_train_valid_dataset.py /path/to/context.json /path/to/train.json /path/to/valid.json
   ```

2. MC training using hfl/chinese-lert-base
    ```bash
    python run_swag_no_trainer.py \
        --model_name_or_path hfl/chinese-lert-base \
        --train_file "dataset_train.csv" \
        --validation_file "dataset_valid.csv" \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ./output_hfl/chinese-lert-base 
    ```
    Here's some models' accuracy with the same training parameters:
    | Model                     | Epoch | Accuracy              |
    |----------------------------|-------|-----------------------|
    | bert-base-chinese           | 2     | 0.9541                |
    | hfl/chinese-lert-base       | 2     | 0.9644                |
    | chinese-macbert-base        | 2     | 0.9634                |
    | hfl/chinese-bert-wwm-ext    | 2     | 0.9588                |



3. QA training using hfl/chinese-lert-base
    ```bash
    python run_qa_no_trainer.py \
        --model_name_or_path hfl/chinese-lert-base \
        --train_file ./qa_train_dataset.csv \
        --validation_file ./qa_valid_dataset.csv \
        --learning_rate 1.7e-5 \
        --num_train_epochs 5 \
        --output_dir ./output_qa/hfl/chinese-lert-base \
        --per_device_eval_batch_size 6  \
        --per_device_train_batch_size 6
    ```
    QA results:
    ```bash
    "eval_exact_match": 83.48288467929545,
    "eval_f1": 83.48288467929545
    ```

    To train from scratch, you can run this command:
    ```bash
    python run_qa_no_trainer.py \
        --model_type bert  \
        --tokenizer_name hfl/chinese-bert-wwm-ext \
        --train_file ./qa_train_dataset.csv \
        --validation_file ./qa_valid_dataset.csv \
        --learning_rate 1.7e-5 \
        --num_train_epochs 5 \
        --output_dir ./output_qa/hfl/chinese-bert-wwm-ext \
        --per_device_eval_batch_size 6  \
        --per_device_train_batch_size 6
    ```
    To train ```chinese-bert-wwm-ext```, you can run this command:
    ```bash
    python run_qa_no_trainer.py \
        --model_name_or_path hfl/chinese-bert-wwm-ext \
        --train_file ./qa_train_dataset.csv \
        --validation_file ./qa_valid_dataset.csv \
        --learning_rate 1.7e-5 \
        --num_train_epochs 5 \
        --output_dir ./output_qa/hfl/chinese-bert-wwm-ext \
        --per_device_eval_batch_size 6  \
        --per_device_train_batch_size 6
    ```
#### Testing

1. Run the ```run.sh``` to do the prediction on testing dataset.
    ```bash
    bash run.sh path/to/context.json path/to/test.json path/to/output.csv
    ```
    Or you can do MC or QA respectively by ```infer_mc.py``` and ```infer_qa.py```

#### Download My Pre-trained model and dataset

Run ```download.sh```