# Course_Capstone_Project
'Capstone Design' course Project


# Environment Installation in Seraph

※You must have Kyung Hee University ID

## How to use seraph
### First. Sign up for Seraph
Access the homepage and sign up.

Seraph url : [Seraph][url]

[url]: http://seraph.khu.ac.kr:52080/ "Seraph"

### Second. Accessing Seraph
1.  ssh [username@address | ex: username@moana.khu.ac.kr or username@163.180.160.104] -p 30080

2.  ssh [hostname] -p 30080, The following process is required.

\~/.ssh/config(~, Window : C:\Users\username, Linux:/home/username)
<pre>
Host [hostname]
    HostName [address]
    User [username]
</pre>

### After first connecting to use conda, do the following.
<pre>
/data/opt/anaconda3/bin/conda init
exit
</pre>

### See below for the process of using Seraph using sftp.

Tutorials >> 1. Practice

Seraph Manual : [Manual][url2]

[url2]: https://nonstop-gravity-18d.notion.site/SERAPH-KHU-GPU-Cluster-User-Guide-a26618b911ee4e709e85fbe7f4cec807 "Manual"


## Use yaml files to create a virtual environment.

※A virtual environment must be created in /data/[username]/ to use the environment on a compute node.

Two way to install your virtual environment in the desired location.

### First
<pre>
conda config --append envs_dirs [env_PATH | ex: /data/[username]/]
conda env create --file [FILE_PATH] -n [env_Name]
</pre>

### Second
<pre>conda env create --file [FILE_PATH] -p [env_PATH/env_Name]</pre>

The following error occurs with other commands.
<pre>CondaValueError: could not parse 'name: [env_name]' in: [env_File]</pre>

# installed package
## pytorch1.12.1_p38_SAM_window.yaml python==3.10.9
### Installed by conda
<pre>
matplotlib=3.7.1
numpy=1.23.5
opencv=4.6.0
pandas=1.5.2
python=3.10.9
pytorch=2.0.1=py3.10_cuda11.7_cudnn8_0
scikit-image=0.19.3
scikit-learn=1.1.2
segment-anything=1.0
torchaudio=2.0.2
torchvision=0.15.2
</pre>

## pytorch1.12.1_p38_SAM_linux.yaml python=3.8.13

Not include opencv-python. must install opencv-python following command.
<pre>pip install opencv-python==4.6.0.66</pre>

### Installed by conda
<pre>
cudatoolkit=11.3.1
matplotlib=3.6.0
numpy=1.24.3
pandas=1.4.4
python=3.8.13
pytorch=1.12.1=py3.8_cuda11.3_cudnn8.3.2_0
scikit-image=0.19.3
scikit-learn=1.1.2
segment-anything=1.0
tar=1.34
torchaudio=0.12.1
torchvision=0.13.1
</pre>

## pytorch1.12.1_p38_SAM_opencv_linux.yaml python==3.8.13

### Installed by conda
<pre>
cudatoolkit=11.3.1
matplotlib=3.6.0
numpy=1.24.3
pandas=1.4.4
python=3.8.13
pytorch=1.12.1=py3.8_cuda11.3_cudnn8.3.2_0
scikit-image=0.19.3
scikit-learn=1.1.2
segment-anything=1.0
tar=1.34
torchaudio=0.12.1
torchvision=0.13.1
</pre>

### installed by pip
<pre>
opencv=4.6.0.66
</pre>