# Counterfeit Currency Detection using Image Processing

This repository contains a modular, notebook-driven counterfeit currency detection system designed to inspect 12 distinct visual, geometric, and chromatic security features on Indian Rupee banknotes (Rs. 500 and Rs. 2000). The workflow is orchestrated by a central controller that executes a starter GUI, routes to the denomination-specific analysis pipeline, and renders a comprehensive validation report GUI.

For a detailed walkthrough of the design, state variables, and algorithmic thresholds, refer to the [Study Guide](docs/study-guide.md) and the [Technical Documentation](docs/project-documentation.md).

---

## Workspace Layout

Following a professional, academic restructuring, the workspace layout is organized as follows:

- **`notebooks/`**: Contains the interactive user interfaces (`gui_1.ipynb`, `gui_2.ipynb`), denomination testing notebooks (`500_Testing.ipynb`, `2000_Testing.ipynb`), and the execution orchestrator (`controller.ipynb`).
- **`src/`**: Houses the headless batch test runner (`evaluate.py`), active template generator (`augment_dataset.py`), and simulated counterfeit generator (`synthesize_fakes.py`).
- **`data/`**: Structured repository separating real genuine banknotes, real physical counterfeits, programmatically synthesized fakes, and reference template crop folders.
- **`assets/`**: Fallback assets such as missing crop indicators.
- **`scripts/`**: Archived one-off analysis and debug scripts.
- **`docs/`**: Project proposal and technical documentation.

---

## Quick Start

### 1. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Active Template Data Augmentation
To expand reference template matches and maximize keypoint extraction success under varying camera angles, run the augmentation utility:
```bash
python src/augment_dataset.py
```

### 3. Launch the Application
1. Navigate to the `notebooks/` directory and open `controller.ipynb` using Jupyter Notebook or JupyterLab.
2. Execute all cells in order.
3. Use the starting Tkinter GUI to browse for a banknote image (`.jpg`) and select the denomination.
4. The system will process the note and automatically render the report GUI showing the per-feature verdicts.

### 4. Headless Quantitative Evaluation
To run the evaluation pipeline in headless batch mode and generate accuracy, precision, and F1 metrics for both real-world banknotes and synthetic probes:
```bash
python src/evaluate.py
```

---

## Dataset Credit

Dataset credit goes to **aprameya2001**.
