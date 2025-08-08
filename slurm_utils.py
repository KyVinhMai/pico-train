import os
import Path
import yaml


#Provide option to train on UCI HPC3, local hardware, or Google Colab Cloud

#Very hardcoded

def generate_batch_script(config_path):
    """Generate a SLURM batch script for LLaMA experiments"""


    # model_size="13",
    # prompt_type="simplified_prompt", 
    # temperatures=[0.1, 0.5, 0.9],
    # scale_ranges=["1.00"],
    # account="RFUTRELL_LAB_GPU",
    # partition="gpu",
    # gpu_type="A100",
    # gpu_count=1,
    # cpus=4,
    # memory="64G",
    # time="12:00:00",
    # job_name=None,
    output_dir="batch_scripts"

    config = yaml.safe_load(open(config_path, "r"))

    s = config.slurm
    
    
    # Template for the batch script
    batch_template = f"""#!/bin/bash
#SBATCH --job-name={s.job_name}
#SBATCH --partition={s.partition}
#SBATCH --account={s.account}
#SBATCH --gres=gpu:{s.gpu_type}:{s.gpu_count}
#SBATCH --cpus-per-task={s.cpus}
#SBATCH --mem={s.memory}
#SBATCH --time=12:00:00
#SBATCH --output=logs/{s.job_name}_%j.out
#SBATCH --error=logs/{s.job_name}_%j.err

# Load required modules
module load python/3.10.2
module load cuda/11.7.1

# Activate conda environment
source ~/.bashrc
conda activate pretraining_project

# Set CUDA environment variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0

cp {config_path} $TMPDIR/
cp -r src/ $TMPDIR/
cd $TMPDIR

# Add current directory to PYTHONPATH
export PYTHONPATH=$TMPDIR:$PYTHONPATH
    
poetry run train --config_path custom-gpt2-Amino-Acids.yaml


echo "Pretraining Job completed at $(date)"
"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate filename
    filename = f"run_job.sh"
    filepath = Path(output_dir) / filename
    
    # Write the batch script
    with open(filepath, 'w') as f:
        f.write(batch_template)
    
    # Make it executable
    os.chmod(filepath, 0o755)
    
    print(f"Generated batch script: {filepath}")
    return filepath


def submit_job(script_path, dry_run=False):
    """Submit a batch script to SLURM"""
    import subprocess
    
    if dry_run:
        print(f"DRY RUN: Would submit {script_path}")
        return None
    
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Extract job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted {script_path} -> Job ID: {job_id}")
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {script_path}: {e}")
        print(f"STDERR: {e.stderr}")
        return None