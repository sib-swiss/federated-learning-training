## 1. Install Conda

If you do not already have **Conda** installed, we recommend installing **Miniconda**, a lightweight version of Anaconda.

1. Go to the [Anaconda download page](https://www.anaconda.com/download) and sign in or create an account if prompted.
2. Open the **Distribution** section and download **Miniconda** for Linux.
3. Run the installer in your terminal:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

4. Follow the instructions and restart your terminal when done.

To verify the installation, run:

```bash
conda --version
```

---

## 2. Install Git

If you do not already have **Git** installed, use your distribution's package manager.

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install git
```

**Fedora:**

```bash
sudo dnf install git
```

**Arch:**

```bash
sudo pacman -S git
```

---

## 3. Install VS Code

If you do not already have **Visual Studio Code** installed, please install it as your editor for the course.

1. Go to the [VS Code download page](https://code.visualstudio.com/) and download the installer for Linux (`.deb` for Ubuntu/Debian, `.rpm` for Fedora).
2. Follow the installation instructions and launch VS Code once complete.

---

## 4. Clone the Course Repository

1. Open **VS Code** and click **Clone Git Repository** on the start screen.
2. Enter the repository URL:

```text
https://github.com/sib-swiss/federated-learning-training.git
```

3. Choose a folder where you would like to save the course files and click **Open** once the download is complete.

---

## 5. Set Up the Environment

We provide a Conda environment file that installs all required packages for the course.

1. Open **VS Code** and load the cloned `federated-learning-training` folder.
2. Open a terminal via **Terminal > New Terminal** and run:

```bash
conda env create -f fl-course-env.yaml
```

This may take a few minutes.

3. Once complete, activate the environment:

```bash
conda activate fl-course-env
```

4. Select the `fl-course-env` interpreter in VS Code if prompted.

Once activated, your terminal prompt should look like this:

```bash
(fl-course-env) your-name@your-computer federated-learning-training %
```

