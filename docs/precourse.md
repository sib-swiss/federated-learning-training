# Course Prerequisites

To ensure everything runs smoothly during the course, please complete the steps below **before arriving**. This setup is essential so that all required software and packages are ready to use, allowing us to start the hands-on exercises immediately and make the best use of our time together.

Choose your operating system below:

=== "macOS"

    ## For macOS

    ### 1. Install Conda

    If you do not already have **Conda** installed, we recommend installing **Miniconda**, a lightweight version of Anaconda.

    #### Steps

    1. Go to the [Anaconda download page](https://www.anaconda.com/download).
    2. Sign in or create an account if prompted.
    3. Open the **Distribution** section.
    4. Download **Miniconda** for your operating system.
    5. Run the installer and follow the instructions.

    To verify that Conda was installed correctly, open a terminal and run:

    ```bash
    conda --version
    ```

    This should display the installed Conda version number.

    ### 2. Install VS Code

    If you do not already have **Visual Studio Code (VS Code)** installed, please install it as your editor for the course.

    #### Steps

    1. Go to the official [VS Code download page](https://code.visualstudio.com/).
    2. Download the installer for your operating system (Windows, macOS, or Linux).
    3. Follow the installation instructions for your platform.
    4. Launch VS Code once installation is complete.

    To verify that VS Code was installed correctly, open the application successfully.

    ### 3. Clone the Course Repository

    You will need a local copy of the course GitHub repository to access the course materials and the environment file.

    #### Steps

    1. Open **VS Code**.
    2. On the start screen, click **Clone Git Repository**.
    3. When prompted, enter the repository URL:

    ```text
    https://github.com/sib-swiss/federated-learning-training.git
    ```

    4. Choose the folder where you would like to save the course files.
    5. Once the download is complete, click **Open** to open the repository in VS Code.

    ### 4. Set Up the Environment

    We provide a Conda environment file that installs all required packages for the exercises.

    #### Steps

    1. If it is not already open, open **VS Code** and load the cloned `federated-learning-training` folder.
    2. Open a terminal in VS Code via **Terminal > New Terminal**.
    3. Run the following command:

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

=== "Windows"

    ## For Windows

    ### 1. Install Conda

    If you do not already have **Conda** installed, install **Miniconda** for Windows from the [Anaconda download page](https://www.anaconda.com/download).

    #### Steps

    1. Download the Windows installer.
    2. Run the installer and follow the prompts.
    3. Accept the default settings unless you need a custom location.

    To verify that Conda was installed correctly, open PowerShell or Command Prompt and run:

    ```powershell
    conda --version
    ```

    This should display the installed Conda version number.

    ### 2. Install VS Code

    If you do not already have **Visual Studio Code (VS Code)** installed, please install it as your editor for the course.

    #### Steps

    1. Go to the official [VS Code download page](https://code.visualstudio.com/).
    2. Download the installer for Windows.
    3. Install VS Code and launch it when finished.

    ### 3. Clone the Course Repository

    You will need a local copy of the course GitHub repository to access the course materials and the environment file.

    #### Steps

    1. Open **VS Code**.
    2. On the start screen, click **Clone Git Repository**.
    3. When prompted, enter the repository URL:

    ```text
    https://github.com/sib-swiss/federated-learning-training.git
    ```

    4. Choose the folder where you would like to save the course files.
    5. Once the download is complete, click **Open** to open the repository in VS Code.

    ### 4. Set Up the Environment

    We provide a Conda environment file that installs all required packages for the exercises.

    #### Steps

    1. Open a terminal in VS Code via **Terminal > New Terminal**.
    2. Run the following command:

    ```powershell
    conda env create -f fl-course-env.yaml
    ```

    This may take a while.

    3. Once the installation is complete, activate the environment:

    ```powershell
    conda activate fl-course-env
    ```

    4. In VS Code, select the `fl-course-env` interpreter if prompted.

    Once activated, your terminal prompt should show the environment name at the beginning of the command line.

=== "Linux"

    ## For Linux

    ### 1. Install Conda

    If you do not already have **Conda** installed, install **Miniconda** for Linux from the [Anaconda download page](https://www.anaconda.com/download).

    #### Steps

    1. Download the Linux installer.
    2. Run the installer and follow the instructions.
    3. You may need to make the downloaded file executable before running it.

    To verify that Conda was installed correctly, open a terminal and run:

    ```bash
    conda --version
    ```

    This should display the installed Conda version number.

    ### 2. Install VS Code

    If you do not already have **Visual Studio Code (VS Code)** installed, please install it as your editor for the course.

    #### Steps

    1. Go to the official [VS Code download page](https://code.visualstudio.com/).
    2. Download the installer for Linux.
    3. Follow the installation instructions for your distribution.

    ### 3. Clone the Course Repository

    You will need a local copy of the course GitHub repository to access the course materials and the environment file.

    #### Steps

    1. Open **VS Code**.
    2. On the start screen, click **Clone Git Repository**.
    3. When prompted, enter the repository URL:

    ```text
    https://github.com/sib-swiss/federated-learning-training.git
    ```

    4. Choose the folder where you would like to save the course files.
    5. Once the download is complete, click **Open** to open the repository in VS Code.

    ### 4. Set Up the Environment

    We provide a Conda environment file that installs all required packages for the exercises.

    #### Steps

    1. Open a terminal in VS Code via **Terminal > New Terminal**.
    2. Run the following command:

    ```bash
    conda env create -f fl-course-env.yaml
    ```

    This may take a while.

    3. Once the installation is complete, activate the environment:

    ```bash
    conda activate fl-course-env
    ```

    4. In VS Code, select the `fl-course-env` interpreter if prompted.

    Once activated, your terminal prompt should show the environment name at the beginning of the command line.
