# !/bin/sh 
## General options 
## -- specify queue -- 
# BSUB -q hpc
## -- set the job Name -- 
# BSUB -J tymt_Application
## -- ask for number of cores (default: 1) -- 
# BSUB -n 4
## -- specify that the cores must be on the same host -- 
# BSUB -R "span[hosts=1]"
## -- specify that we need 4GB of memory per core/slot -- 
# BSUB -R "rusage[mem=4GB]"
## -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
# BSUB -M 5GB
## -- set walltime limit: hh:mm -- 
# BSUB -W 6:00 
## -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
# BSUB -u s241485@student.dtu.dk
### -- send notification at start -- 
# BSUB -B 
### -- send notification at completion -- 
# BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
# BSUB -o low-light-enhancement/pix2pix/outputs/
# BSUB -e low-light-enhancement/pix2pix/errors/

module load python3/3.12.1
python3 -m pip install --user tensorflow
python3 -m pip install --user matplotlib
python3 -m pip install --user ipython
python3 -m pip install --user opencv-python
python3 -m pip install --user imageio
python3 -m pip install --user colour-science
python3 -m pip install --user pandas
python3 -m pip install --user tensorboard

# here follow the commands you want to execute with input.in as the input file
python3 low-light-enhancement/pix2pix/pix2pix.py