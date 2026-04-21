
## 1. Enable long Path in windows

### Steps

1. Open the terminal with admin rights
2. Run the following command:

```bash
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```
3. Restart the pc


## 2. Install Conda
 
If you do not already have **Conda** installed, we recommend installing **Miniconda**, a lightweight version of Anaconda.
 
### Steps
 
1. Go to the [Anaconda download page](https://www.anaconda.com/download).
2. Sign in or create an account if prompted.
3. Open the **Distribution** section.
4. Download **Miniconda** for your operating system.
5. Run the installer and follow the instructions. During installation the checkbox **Add miniconda to PATH must be toggled**.
 
To verify that Conda was installed correctly, open a terminal and run:
 
```bash
conda init powershell
conda --version
```
 
This should display the installed Conda version number.
 
---
 
## 3. Install Git
If you do not already have **Git** installed, please install it.
 
### Steps
 
1. Go to the official [Git download page](https://git-scm.com/install/windows).
2. Download the installer.
3. Follow the installation instructions for your platform.

---
 
## 4. Install VS Code
 
If you do not already have **Visual Studio Code (VS Code)** installed, please install it as your editor for the course.
 
### Steps
 
1. Go to the official [VS Code download page](https://code.visualstudio.com/).
2. Download the installer for your operating system
3. Follow the installation instructions for your platform.
4. Launch VS Code once installation is complete.
 
To verify that VS Code was installed correctly, open the application successfully.
 
---
 
## 5. Clone the Course Repository
 
You will need a local copy of the course GitHub repository to access the course materials and the environment file.
 
### Steps
 
1. Open **VS Code**.
2. On the start screen, click **Clone Git Repository**.
3. When prompted, enter the repository URL:
 
```text
https://github.com/sib-swiss/federated-learning-training.git
```
 
4. Choose the folder where you would like to save the course files.
5. Once the download is complete, click **Open** to open the repository in VS Code.
 
---
 
## 6. Set Up the Environment
 
We provide a Conda environment file that installs all required packages for the exercises.
 
### Steps
 
1. If it is not already open, open **VS Code** and load the cloned `federated-learning-training` folder.
2. Open a terminal in VS Code via **Terminal > New Terminal**
3. Run the following command
 
```bash
conda env create -f fl-course-env.yaml
```
 
This may take a while.
 
4. Once the installation is complete, activate the environment:
 
```bash
conda activate fl-course-env
```
 
5. In VS Code, select the `fl-course-env` interpreter if prompted.
 
Once activated, your terminal prompt should show the environment name at the beginning of the command line, for example:
 
```bash
(fl-course-env) your-name@your-computer federated-learning-training %
```