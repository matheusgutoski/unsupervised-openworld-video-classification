#python incremental_main.py --gpu 1 --seed 8 --ti3d_type=gold --online;
#python incremental_main.py --gpu 1 --seed 8 --ti3d_type=incremental --online;
#python incremental_main.py --gpu 1 --seed 8 --ti3d_type=incremental;
#python incremental_main.py --gpu 1 --seed 9 --ti3d_type=incremental;
#python incremental_main.py --gpu 1 --seed 9 --ti3d_type=incremental --online;
#python incremental_main.py --gpu 1 --seed 9 --ti3d_type=gold;
#python incremental_main.py --gpu 1 --seed 9 --ti3d_type=gold --online;
#python incremental_main.py --gpu 1 --seed 9 --ti3d_type=fixed;
#python incremental_main.py --gpu 1 --seed 5 --ti3d_type=incremental --online;
#python incremental_main.py --gpu 1 --seed 5 --ti3d_type=gold --online;



python incremental_main.py --gpu 1 --seed 5 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 21 --incremental_n_classes 20;
python incremental_main.py --gpu 1 --seed 6 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 21 --incremental_n_classes 20;
python incremental_main.py --gpu 1 --seed 7 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 21 --incremental_n_classes 20;
python incremental_main.py --gpu 1 --seed 8 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 21 --incremental_n_classes 20;
python incremental_main.py --gpu 1 --seed 9 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 21 --incremental_n_classes 20;


python incremental_main.py --gpu 1 --seed 5 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 6 --incremental_n_classes 5;
python incremental_main.py --gpu 1 --seed 6 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 6 --incremental_n_classes 5;
python incremental_main.py --gpu 1 --seed 7 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 6 --incremental_n_classes 5;
python incremental_main.py --gpu 1 --seed 8 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 6 --incremental_n_classes 5;
python incremental_main.py --gpu 1 --seed 9 --ti3d_type=incremental --online --incremental_evm --tail_size 0.2 --cover_threshold 0.99 --initial_n_classes 6 --incremental_n_classes 5;




