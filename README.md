# Research of Automated Pathology Report Generation from Whole-Slide Histopathology Image
Research for my master’s thesis \
Special thanks to 成大醫院林鵬展醫師、暨大資工周信宏老師和成大資工謝孫源老師

# Dataset
Data used in the research are collected from NCKU hospital \
**Due to privacy concerns, the dataset cannot be made publicly available.** \
Please feel free to contact us via email for further discussion if needed.
# Motivation
* Issue of image data: WSI has large size and high resolution large amount of data digh computational resource few patch-level label, labeling is labor-intensive and time-consuming
* Issue of texture data: diversity report description It is difficult for the model to learn different doctors may subjectively emphasize distinct features.

>**Objective**: Develop an integrated framework that combines WSI analysis and report generation to enhance diagnostic efficiency and reduce pathologists’ workload.


# Framework Overview
![Workflow](Imgs/Workflow_v2.png)

# Report generation example
![EX1](Imgs/generation_ex_1.png)
![EX2](Imgs/generation_ex_2.png)

# Attention heatmap
![Heatmap](Imgs/Attention_map_ex1.png)