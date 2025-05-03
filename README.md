# Bar Exam Model Bias Analysis Dashboard
**Note** : The result data from inference runs that is to be analysed, can be downloaded from [this Huggingface LFS repo](https://huggingface.co/datasets/HolySaint/mcq_bias_analysis_data/blob/main/ALL_llama_new_question_level_analysis_results.csv) 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://model-accuracy-analyser-9rjrrn7fnmpnngw9jxx7re.streamlit.app/)This Streamlit appprovides an interactive dashboard focused on analyzing potential biases in model answer selections for multiple-choice questions. It's designed to explore how different model configurations might favor certain answer options (A, B, C, D) compared to the ground truth distribution. I  created and used this in my paper [A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam](https://arxiv.org/abs/2504.04945) as a supplementary tool to analyse the effect of supervised fine-tuning on the inherent preference towards certain options, and how SFT mitigates that.

## Overview

The dashboard allows users to investigate potential biases by:

1.  **Analyzing Option Selection Bias:** Comparing the distribution of predicted answer options against the distribution of correct answers, both overall and grouped by specific experimental parameters.
2.  **Analyzing Label Confusion:** Visualizing how often the model confuses one correct answer label for another, helping identify systematic error patterns.

## Features

* **Data Loading:** Loads pre-processed, question-level analysis data from a specified CSV file path (currently hardcoded, can be modified).
* **Configuration Selection:** Focus the analysis on a specific model (`model_name`) and training status (`trained`/`untrained`) using simple selection widgets.
* **Flexible Filtering:** Apply additional filters based on parameters like generation strategy, response format, prompt type, etc.
* **Option Selection Bias Analysis:**
    * Visualize overall predicted vs. ground truth option distributions (A, B, C, D).
    * Quantify overall bias using Root Mean Square Error (RMSE).
    * Optionally group the bias analysis by **one** additional parameter (e.g., `generation`, `response_type`) to see how it influences option selection tendencies.
    * Analyze bias conditioned on the ground truth label (e.g., "When the answer is 'A', how biased is the model's prediction distribution?").
* **Label Confusion Analysis:**
    * Display an overall confusion matrix showing prediction patterns based on the ground truth label.
    * Visualize label accuracy grouped by the selected parameter.
* **Exporting:** Download plots as PNG, HTML, or JSON.

## Expected Data Format

The application expects a CSV file (like `ALL_llama_new_question_level_analysis_results.csv`) with question-level results, including columns such as:

* `model_name`: Identifier for the model (e.g., 'llama2', 'llama3').
* `training_status`: 'trained' or 'untrained'.
* `predicted_label`: The option (A, B, C, D) predicted by the model.
* `ground_truth_label`: The correct option (A, B, C, D).
* Other potential filtering/grouping columns: `generation`, `response_format`, `response_type`, `prompt_type`, `explanation_type`, `dataset`, `eval_dataset`.
* `file_exists` (Optional): Used for initial filtering if present.

*Note: The script assumes the presence of `model_name`, `training_status`, `predicted_label`, and `ground_truth_label`.*

## Setup and Usage for local running

1.  **Prerequisites:**
    * Python 3.8+
    * pip (Python package installer)

2.  **Clone the Repository (Optional):**
    ```bash
    git clone https://github.com/ReanFernandes/mcq-bias-analyser.git
    cd your-repo-name
    ```


3.  **Install Dependencies:**
    It's recommended to create a virtual environment:
    ```bash
    # Create virtual environment
    python -m venv venv
    # Activate it (Linux/macOS)
    source venv/bin/activate
    # Or activate it (Windows)
    # venv\Scripts\activate

    # Install required packages
    pip install -r requirements.txt
    ```

4.  **Run the App:**
    Navigate to the directory containing the script and run:
    ```bash
    streamlit run response_bias_analysis.py
    ```


5.  **Using the App:**
    * Open the URL provided by Streamlit (usually `http://localhost:8501`) in your browser.
    * Use the sidebar to:
        * Select the specific `Model Name` and `Training Status` to analyze.
        * Apply additional filters if needed.
        * Optionally select ONE parameter to group the analysis by.
        * Choose a color palette.
    * Navigate between the "Option Selection Bias" and "Label Confusion Analysis" tabs.
    * Use the "Customize Plot Appearance" expander to fine-tune visualizations.
    * Download plots using the export buttons.

## Dependencies

The main libraries used are:

* Streamlit
* Pandas
* Plotly
* NumPy
* SciPy (used implicitly by other libraries, good practice to include)



