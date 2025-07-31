# RGE Classifiers Repository

This repository contains the code used for fine-tuning the ModernBERT model on the RGE (Really Good Emails) dataset for category classification.

## Repository Structure

```
RGE_Classifiers/
│
├── Binary/
│   ├── experiment_configuration.json   # Experiment configuration for binary classification
│   └── train.py                       # Training script for binary classification
│
├── MultiLabel/
│   ├── categoriesList.json            # List of categories for multilabel classification
│   ├── experiment_configuration.json  # Experiment configuration for multilabel classification
│   └── train.py                       # Training script for multilabel classification
```

### Folder Description

- **Binary/**: contains code and configuration for binary email classification.
- **MultiLabel/**: contains code, configuration, and the list of categories for multilabel email classification.

### Usage


1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit the configuration files (`experiment_configuration.json`) as needed.
3. Run the training scripts (`train.py`) in the respective folders to train the models.


### Notes
- The repository is designed to be easily extendable to other classification tasks.
- All required dependencies are listed in `requirements.txt`.
- The dataset used for training (RGE - Really Good Emails) is not included in this repository. To obtain the data, you must request access from the authors of the original paper.

For questions or contributions, please open an issue or a pull request.


# Proprietary Code – © 2025 Growens, Inc.
# All rights reserved.
#
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, modification, or use
# of this code, in whole or in part, is strictly prohibited
# without prior written permission from the owner.