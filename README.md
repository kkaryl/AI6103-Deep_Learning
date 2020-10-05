# AI6103_2020
Master of AI, Deep learning course AI6103, 2020


<br><br>


### Cloud Machine #1 : Google Colab (Free GPU)

* Follow this Notebook installation :<br>
https://colab.research.google.com/github/xbresson/AI6103_2020/blob/master/codes/installation/installation.ipynb

* Open your Google Drive :<br>
https://www.google.com/drive

* Open in Google Drive Folder 'AI6103_2020' and go to Folder 'AI6103_2020/codes/'<br>
Select the notebook 'file.ipynb' and open it with Google Colab using Control Click + Open With Colaboratory



<br><br>

### Cloud Machine #2 : Binder (No GPU)

* Simply [click here]

[Click here]: https://mybinder.org/v2/gh/xbresson/AI6103_2020/master




<br><br>

### Local Installation for OSX & Linux

* Open a Terminal and type


```sh
   # Conda installation
   curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh # Linux
   curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # OSX
   chmod +x ~/miniconda.sh
   ./miniconda.sh
   source ~/.bashrc

   # Clone GitHub repo
   git clone https://github.com/xbresson/AI6103_2020.git
   cd AI6103_2020

   # Install python libraries
   conda env create -f environment.yml
   source activate deeplearn_course

   # Run the notebooks
   jupyter notebook
   ```




### Local Installation for Windows 

```sh
   # Install Anaconda 
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

   # Open an Anaconda Terminal 
   Go to Application => Anaconda3 => Anaconda Prompt 

   # Install git : Type in terminal
   conda install git 

   # Clone GitHub repo
   git clone https://github.com/xbresson/AI6103_2020.git
   cd AI6103_2020

   # Install python libraries
   conda env create -f environment_windows.yml
   source activate deeplearn_course

   # Run the notebooks
   jupyter notebook
   ```







<br><br><br><br><br><br>