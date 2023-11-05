export PYTHONPATH="${HOME}/code/CVD"

# nohup python -u run.py > ../log/language/biobert_ds0_with_linear_warmup.log 2>&1 &

nohup python -u run.py > ../../log/language/biobert_ds_ECER_default_full.log 2>&1 &

# python run.py
