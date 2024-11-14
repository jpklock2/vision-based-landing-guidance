import argparse
import os
from trackit.core.boot.main import main


def setup_arg_parser():
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('--method_name', default='LoRAT', type=str, help='Method name')
    parser.add_argument('--config_name', default='dinov2', type=str, help='Config name')
    parser.add_argument('--output_dir', help='path where to save')
    parser.add_argument('--dry_run', action='store_true', help='do not save checkpoints and results')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--instance_id', type=int)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--quiet', action='store_true', help='do not generate unnecessary messages.')

    parser.add_argument('--pin_memory', action='store_true', help='move tensors to pinned memory before transferring to GPU')

    parser.add_argument('--disable_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--wandb_run_offline', action='store_true', help='run wandb offline')
    parser.add_argument('--enable_stack_trace_on_error', action='store_true', help='enable stack trace on error')
    parser.add_argument('--allow_non_master_node_print', action='store_true', help='enable logging on non-master nodes')

    parser.add_argument('--do_sweep', action='store_true')
    parser.add_argument('--sweep_config', type=str)

    parser.add_argument('--mixin_config', type=str, action='append')

    parser.add_argument('--run_id', type=str)

    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--distributed_node_rank', type=int, default=0)
    parser.add_argument('--distributed_nnodes', type=int, default=1)
    parser.add_argument('--distributed_nproc_per_node', type=int, default=1)
    parser.add_argument('--distributed_do_spawn_workers', action='store_true')

    parser.add_argument('--wandb_distributed_aware', action='store_true')
    parser.add_argument('--kill_other_python_processes', action='store_true')
    parser.add_argument('--multiprocessing_start_method_spawn', action='store_true')
    parser.add_argument('--weight_path', type=str, action='append')

    return parser

def main_tracking():

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))

    args.method_name = "LoRAT"
    args.config_name = "dinov2"
    args.mixin_config = "evaluation"
    args.mixin_config = "my_dataset_test"
    args.weight_path = "/mnt/d/Master/Airport_Runway_Detection/LARD/data/LoRAT_output/LoRAT-dinov2-mixin-my_dataset_train-2024.08.15-04.05.01-266394/LoRAT-dinov2-mixin-my_dataset_train-2024.08.15-04.05.01-266394/checkpoint/epoch_21/model.bin"
    args.run_id = "LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434"
    args.output_dir = "/mnt/d/Master/Airport_Runway_Detection/vision-based-landing-guidance/LoRAT/output/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434"
    args.wandb_run_offline = True
    args.disable_wandb = True

    main(args)

    #   |& tee -a /mnt/d/Master/Airport_Runway_Detection/LARD/data/LoRAT_output/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434/train_stdout.log
    # PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 python main.py LoRAT dinov2 --run_id LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434 --output_dir /mnt/d/Master/Airport_Runway_Detection/LARD/data/LoRAT_output/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434 --wandb_run_offline --disable_wandb --weight_path /mnt/d/Master/Airport_Runway_Detection/LARD/data/LoRAT_output/LoRAT-dinov2-mixin-my_dataset_train-2024.08.15-04.05.01-266394/LoRAT-dinov2-mixin-my_dataset_train-2024.08.15-04.05.01-266394/checkpoint/epoch_21/model.bin --mixin_config my_dataset_test --mixin_config evaluation |& tee -a /mnt/d/Master/Airport_Runway_Detection/LARD/data/LoRAT_output/LoRAT-dinov2-mixin-my_dataset_test-mixin-evaluation-2024.11.14-11.12.11-742434/train_stdout.log


if __name__ == '__main__':

    main_tracking()