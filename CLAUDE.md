
# CLAUDE.md  - Project/Paper Guide

## Project Overview
This project uses the **C3-LDM (Census-Consistent Conditional Latent Diffusion Model)** to generate high-resolution population density maps by integrating **VIIRS Nightlights** and **WSF Settlement Masks**. The model uses a residual diffusion approach, supports multiple datasets (WorldPop, GHS-POP, HRSL), and ensures census total consistency for each administrative unit.

### Project Goals:
- Generate high-resolution population density maps via deep learning models
- Address registration issues between nightlight data and settlement masks
- Train and validate diffusion models, evaluating performance on multiple datasets

---

## 🧑‍💻 How to Run and Modify Code
### 1. Setting Up Environment:
Ensure the dependencies are installed correctly when running in **Colab** and you can successfully execute the following steps.

#### a) Install Dependencies:
The following dependencies are required for this project:
```bash
pip install -r requirements.txt
```

#### b) Download the Dataset:
- You need to download and configure **VIIRS Nightlights** and **WSF Settlement Masks**, which should be placed in the `data/paired_dataset/` folder.

#### c) Running on Colab:
You can directly run the following code in Colab:
```python
!git clone https://github.com/your-repo-link
%cd your-repo-directory
!pip install -r requirements.txt
```

Then, follow the instructions in `train.py` or `inference.py` for training and inference.

---

### 2. Main Scripts
- **`train.py`**: The training script, which loads data, defines loss functions and optimizers, and runs the training process.
- **`inference.py`**: The inference script, which generates the high-resolution population maps using the trained model weights.
- **`eval/`**: Contains evaluation scripts that test model performance and generate evaluation metrics.

---

## 🔧 Claude Task Execution Rules

### 1. **Do Not Run Training Tasks**:
- Large-scale training tasks should not be run on **Colab** due to limited computational resources. Claude will assist you by checking the code, fixing small issues, and modifying functionalities, but please avoid running the full training process.
- You can use `@claude fix the failing tests` to have Claude fix issues and ensure the code runs properly.

### 2. **Code Review and Modifications**:
Claude will review code based on your requests. You can use `@claude` in **PR** or **Issue**:
```text
@claude review this PR and suggest improvements
@claude implement feature X
```
Claude will automatically inspect the code and provide suggestions, improve code structure, optimize the training process, or enhance data processing capabilities.

### 3. **Model Modifications**:
You can have Claude modify the model architecture, such as switching to a different baseline model, changing the diffusion process parameters, etc.:
```text
@claude change the baseline model to CNNBaseline
@claude modify the diffusion noise prediction layer
```

### 4. **Fixing Bugs and Failing CI**:
Claude can also help fix bugs or CI errors encountered during the training process:
```text
@claude fix failing tests
```

---

## 📄 Writing the Paper and Generating Reports
Claude will assist with paper writing and help you organize project structure and result analysis.

### 1. **Experiment Design and Result Analysis**:
You can have Claude generate the experiment section and summarize the evaluation results:
```text
@claude write the experiment section including model configurations and results
@claude summarize the evaluation results for the final paper
```

### 2. **Charts and Visualizations**:
Claude can assist in generating experimental charts to visualize model performance and evaluation metrics:
```text
@claude generate the RMSE plot comparing different models
@claude create a table summarizing model evaluation metrics
```

---

## 🔐 Security and Permissions

### 1. **API Key Configuration**:
Claude uses the **Anthropic API Key** securely stored as a GitHub Secret. Ensure this key is correctly set.

### 2. **Repository Access**:
Claude will only have access to repositories you've authorized. Make sure the repository is **private** or **protected**, and the GitHub App permissions are correctly configured to allow necessary modifications to the project.

### 3. **Confidentiality and Compliance**:
Claude will adhere to strict confidentiality agreements and only access the necessary data while executing tasks. All operations are conducted via GitHub Actions and will not expose sensitive information.

---

## ⚡ Expanding the Project Further
If you want Claude to participate in other aspects of the project, you can request it to execute the following tasks in **Colab**:

- Automatic data preprocessing
- Running experiments and evaluating data
- Implementing more advanced models and algorithm optimizations

You can initiate these tasks by creating comments with `@claude` tags.

---

## 🔧 Limitations and Considerations
- Avoid running long training tasks, especially on Colab or other resource-limited environments.
- Claude currently cannot use GPU resources directly, so model training and tuning might need to be performed on local machines or cloud GPU environments.

---

This **CLAUDE.md** file serves to:

1. **Ensure Claude only executes meaningful tasks** within the project to avoid wasting computational resources.
2. **Enable Claude to assist in writing the paper**, automatically generating experiments, charts, and analysis reports.
3. **Guide how to use Claude for code review, bug fixing, and model modifications**, enhancing development efficiency.

---

If you need any modifications or further assistance, feel free to let me know!
