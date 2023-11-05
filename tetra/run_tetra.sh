export PYTHONPATH="${HOME}/code/CVD"
# T1 Discharge_summary
# T2 ECER
nohup python -u main.py > ../log/tetra/T2_single_numlayer_4_toplayer_4_heads_12_relu.log 2>&1 &

# python main.py