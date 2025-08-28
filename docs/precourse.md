To ensure everything runs smoothly during the course, please complete the steps below **before arriving**.

---

## 1. Install Conda

If you don’t already have Conda installed, we recommend installing **Miniconda**, a lightweight version of Anaconda.

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

##  2. Clone the course repository

You will need a local copy of this GitHub repository to access the course materials and the environment file.  

### Steps

1. Open a terminal (or command prompt).  
2. Navigate to the folder where you want to store the course.  
3. Run the following commands:  

```bash
git clone https://github.com/sib-swiss/federated-learning-training.git
cd federated-learning-training
```
##  3. Set up the environment

We provide a Conda environment file that installs all required packages for the exercises.

### Steps
1. From inside the `federated-learning-training` folder, run
```bash
conda env create -f fl-course-env.yaml
```
to create the environment. This may take a while.

2. Activate the environment with
```bash
conda activate fl-course-env
```
Once activated, your terminal prompt should change to show (fl-course-env) at the beginning.

You are now all set up for the course!