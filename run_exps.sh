#!/bin/bash

python nmf_pytorch.py --algo=1 --data_option=1 --max_epochs=1000 &
python nmf_pytorch.py --algo=1 --data_option=2 --max_epochs=1000 &
python nmf_pytorch.py --algo=1 --data_option=3 --max_epochs=1000 &

python nmf_pytorch.py --algo=2 --data_option=2 --beta=0.0 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=2 --beta=0.4 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=2 --beta=0.2 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=1 --beta=0.0 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=1 --beta=0.4 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=1 --beta=0.2 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=3 --beta=0.0 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=3 --beta=0.4 --max_epochs=1000 &
python nmf_pytorch.py --algo=2 --data_option=3 --beta=0.2 --max_epochs=1000 &

python nmf_pytorch.py --algo=3 --data_option=1 --max_epochs=1000 &
python nmf_pytorch.py --algo=3 --data_option=2 --max_epochs=1000 &
python nmf_pytorch.py --algo=3 --data_option=3 --max_epochs=1000 &

python nmf_pytorch.py --algo=4 --data_option=1 --max_epochs=1000 &
python nmf_pytorch.py --algo=4 --data_option=2 --max_epochs=1000 &
python nmf_pytorch.py --algo=4 --data_option=3 --max_epochs=1000 &

