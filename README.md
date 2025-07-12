# Simple Fine Tuning on CPU
This project shows how to fine-tune a lightweight open-source language model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685).
In order to run the code, follow these recommended steps. In our environment, we had Python 3.12 installed.

**Notes:** 
  - This repo is for educational experiments only — results will be minimal on CPU with tiny datasets.
  - For better results, use larger datasets, more training steps, or run on a GPU.
  - The example dataset can be swapped out to test different behaviors.


## Steps to Run the Code

1. **Clone the repository:**
   ```sh
   git clone https://github.com/gitist/simple_fine_tune_on_cpu.git
   cd simple_fine_tune_on_cpu
   ```

2. **Create a python virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment and install dependencies:**
    ```sh
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4. **Run fine tune code:**
    ```sh
    python finetune.py
    ```
5. **Run the inference:**
    ```sh
    python infer.py
    ```
