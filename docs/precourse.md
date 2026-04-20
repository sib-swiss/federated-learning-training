To ensure everything runs smoothly during the course, please complete the steps below **before arriving**.

---

## 1. Install Conda

If you don’t already have **Conda** installed, we recommend installing **Miniconda**, a lightweight version of Anaconda.

### Steps

1. Go to the official [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Follow the platform-specific installation instructions provided on the site.
4. After installation, close and reopen your terminal or command prompt.

To verify that Conda was installed correctly, you can run:

```bash
conda --version
```
This should print the Conda version number.

## 2. Install VS Code

If you don’t already have **Visual Studio Code (VS Code)** installed, please install it as your editor for the course.

### Steps

1. Go to the official [VS Code download page](https://code.visualstudio.com/).
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Follow the platform-specific installation instructions provided on the site.
4. After installation, launch VS Code.

To verify that VS Code was installed correctly, open the application successfully.

## 3. Clone the Course Repository

You will need a local copy of this GitHub repository to access the course materials and the environment file.

### Steps

1. Open **VS Code**.
2. On the start screen, click **Clone Git Repository**.
3. When prompted, enter the repository URL:

```text
https://github.com/sib-swiss/federated-learning-training.git
```
4. Choose the folder where you would like to store the course files.

5. Once the download is complete, click Open to open the repository in VS Code.

## 4. Set Up the Environment

We provide a Conda environment file that installs all required packages for the exercises.

### Steps

1. In **VS Code**, open the cloned `federated-learning-training` folder.
2. Open the **Command Palette** (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS).
3. Search for and select **Python: Create Environment**.
4. Choose **Conda** as the environment type.
5. When prompted, select **Environment file** and choose:

```text
fl-course-env.yaml
```
6. Wait while VS Code creates the environment and installs the required packages. This may take a while.

7. Once completed, select the newly created environment as the active Python interpreter if prompted.

You are then ready to use the course environment in VS Code.