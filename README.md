# Rethinking Multi-domain Generalization with A General Learning Objective, Accepted by CVPR24
Zhaorui Tan, Xi Yang, Kaizhu Huang

arxiv: https://arxiv.org/abs/2402.18853

This repo includes GMDG applied to classification, regression, and segmentation tasks.


Updated 2024/11/18
Hey there, I have a reported issue:
"I noticed a mismatch between the reported results for the TerraInc dataset in the main text and the appendix. Specifically, in Table 7 of the main text, the mean accuracy of the proposed GMDG using ResNet-50 as the oracle model is 51.1%, while in Table 15 of the appendix, it is reported as 50.1%."

I am really sorry for the mismatch in the results. I have checked my raw results, and I need to clarify that the 51.1% in the main paper is correct.
Here are the corrected results for each domain:

TerraIncognita 

 	  	
| Seed     | Location 100  | Location 38	  | Location 43  | Location 46  | Avg.  |
|----------|-------|-------|-------|-------|-------|
| seed 0   | 58.58 | 50.24 | 55.79 | 43.53 | 52.04 |
| seed 1   | 57.18 | 47.99 | 53.94 | 41.77 | 50.22 |
| seed 2   | 67.05 | 43.52 | 55.76 | 37.82 | 51.04 |
| mean     | 60.90 | 47.30 | 55.20 | 41.00 | 51.10 |





