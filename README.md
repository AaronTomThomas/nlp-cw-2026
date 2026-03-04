# BestModel Training and Local Evaluation

The BestModel directory contains the RoBERTa-based classifier, training notebook, and Stage 5 local evaluation artifacts for the "Don't Patronize Me" task. Follow the steps below to recreate the Python environment from `requirements.txt` and run the notebook end-to-end.

## 1. Prerequisites

- Python 3.10+ (repo was developed with Python 3.13)
- Git LFS not required, but ensure that the dataset is present under `BestModel/data/`:
  - `data/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv`
  - `data/dev/dev_semeval_parids-labels.csv`
  - `data/test/task4_test.tsv`
- Optional but recommended: an NVIDIA GPU with CUDA 12.x support.

## 2. Create and Activate the Virtual Environment

From the repo root :

```bash
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file captures every package needed to run the notebooks, including PyTorch, Transformers, and Jupyter.
### (Optional) Register the Kernel for Jupyter

Once the virtual environment is active:

```bash
python -m ipykernel install --user --name bestmodel-env --display-name "bestmodel-env"
```
This makes the environment selectable inside JupyterLab/Notebook.

## 3. Reproduce Training (`train_bestmodel.ipynb`)

1. `cd BestModel`
2. Launch Jupyter (pick whichever interface you prefer):
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `train_bestmodel.ipynb` and ensure the `bestmodel-env` kernel (or the freshly created venv) is selected.
4. Run all cells sequentially:
   - The notebook loads and cleans the dataset.
   - An 80/20 split is produced for training/validation.
   - Training runs for 5 epochs with focal loss + weighted sampling (adjustable via the config cell).
   - The final cells evaluate the best checkpoint, save it under `bestmodel/`, export `threshold.json`, and emit logs to `logs/`.

For a non-interactive run you can execute the notebook via nbconvert:

```bash
jupyter nbconvert --execute train_bestmodel.ipynb --to notebook --inplace
```
### Expected Artifacts

After the notebook completes you should see:

- `bestmodel/` – the exported Hugging Face-compatible checkpoint and tokenizer
- `logs/train_log.csv` & `logs/eval_log.csv` – per-step metrics
- `logs/dev_predictions_stage5.csv` & `logs/test_predictions_stage5.csv` – inference outputs used in Stage 5
## 4. Troubleshooting

- **CUDA not available:** The notebook automatically falls back to CPU. Training will be much slower; reduce epochs or batch size if necessary.
- **Missing dataset files:** Re-download the official "Don't Patronize Me" dataset and place the TSV/CSV files in `BestModel/data/` matching the directory structure above.
- **Kernel not showing up:** Re-run the `ipykernel install` command while the venv is activated, then restart Jupyter.

Following these steps will recreate the environment recorded in `requirements.txt` and allow you to retrain the model plus reproduce the logged metrics and error analysis.
