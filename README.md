# AutoDock-GPU

AutoDock-GPU [(Santos-Martins 2021)](https://pubmed.ncbi.nlm.nih.gov/33403848/) is an OpenCL implementation of the widely used tool AutoDock4 (for the docking of small molecules to macromolecular targets) that leverages the highly parallel architecture of GPU hardware to reduce docking runtime by up to 350-fold with respect to a single-threaded process. In addition, the gradient-based local search method ADADELTA, as well as an improved version of the Solis-Wets random optimiser from AutoDock4 reduces the number of calls to the scoring function that are needed to produce good results.

Here we will discuss the installation and running of this tool on a **Linux**-based operating system.

## Requirements

* An NVIDIA-based GPU system capable of running [CUDA](https://developer.nvidia.com/cuda-toolkit).

## Installation

### AutoDock4 and AutoGrid4

If you already have a dedicated location in which to store tools, navigate to that folder. Otherwise first we will create a new folder within our Documents folder.

To install AutoDock4 and AutoGrid4, you will need administrator privileges.

```bash
# Create the new directory and navigate to it
mkdir ~/Documents/Tools

# Install AutoDock4 and AutoGrid4
sudo apt install autodock
sudo apt install autogrid
```

### MGLTools

Next we need to install MGLTools from the browser using this [link](https://ccsb.scripps.edu/mgltools/downloads/). Choose the GUI installer option for Linux, which has the file name `mgltools_Linux-x86_64_xxx_Install`.

We need to give the installer permission to execute as a program. Right-click the file, switch to the `Permissions` tab, and check the box at the bottom that says `Allow executing file as program`.

<p align="center">
    <img src="./assets/mgltools-as-program.png" width=75%>
</p>

Now we can double-click on the install file, and follow the prompts to install MGTools.

Once you have successfully completed this, we need to add MGTools to the `PATH`. The easiest way to "set and forget" this step is to add this step to our `.bashrc` file (this file runs a given set of instructions every time you start an instance of the terminal).

If you have installed MGLTools in a location other than the home directory, alter the code below as necessary. Also amend the version number if you have installed a different version of MGLTools.

```bash
# Change to the home directory
cd ~

# Add the PATH export instruction to the .bashrc file
# The double greater than sign '>>' appends a new line to the given file
echo 'export PATH=$PATH:/home/{username}/MGLTools-1.5.7/bin' >> ~/.bashrc
echo 'export PATH=$PATH:/home/{username}/MGLtools-1.5.7/MGLToolsPckgs/AutoDockTools' >> ~/.bashrc
```

### CUDA

Head over to the NVIDIA [cuda-toolkit page](https://developer.nvidia.com/cuda-toolkit), click the download link, and select your operating system, system architecture, distribution (e.g. Ubuntu), version number (e.g. 22.04), and the installer type you would like. We will select `deb (local)` here.

<p align="center">
    <img src="./assets/cuda-target-platform.png">
</p>