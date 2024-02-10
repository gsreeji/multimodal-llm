# multimodal-llm

pip install jupyterlab

jupyter notebook --generate-config

sudo vi /etc/systemd/system/jupyterlab.service

[Unit]
Description=JupyterLab
After=syslog.target network.target
[Service]
User=root
Environment="PATH=/opt/conda/envs/pytorch/bin:/opt/conda/condabin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/cuda/include:/usr/libexec/gcc/x86_64-redhat-linux/7:/opt/aws/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/ec2-user/.local/bin:/home/ec2-user/bin"
ExecStart=/opt/conda/envs/pytorch/bin/python -m jupyterlab --notebook-dir=/home/ec2-user/notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
[Install]
WantedBy=multi-user.target


sudo vi /etc/systemd/system/jupyterlab.service
sudo systemctl daemon-reload
sudo systemctl start jupyterlab.service
sudo systemctl status jupyterlab.service
