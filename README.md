# Deep learning-based risk prediction for interventional clinical trials based on protocol design: a retrospective study 
Sohrab Ferdowsi, Julien Knafou, Nikolay Borissov, David Vicente Alvarez, Rahul Mishra, Poorya Amini, Douglas Teodoro
----------------

# Summary
### Background: 
Clinical trials (CTs) constitute a major cost in the drug development cycle, yet their success rate is estimated to be no more than 14%. Among the many factors related to CT failure, the poor design of the protocol itself is considered a major risk. As an alternative to the manual protocol risk assessment, we aimed to investigate deep learning methods to create risk prediction models for CT success estimation.
### Methods: 
This retrospective study used a dataset composed of 360,497 CT protocols publicly available from ClinicalTrials.gov registered from 1990 to 2020. Considering protocol changes reported during the CT execution  and their final status, a retrospective risk assignment method was proposed to label CTs according to low, medium, and high risk levels. Then, two deep learning-based architectures based on transformer and graph neural networks were designed to learn how to infer the ternary risk category. These two architectures were further combined to create an ensemble model and compared against a baseline based on bag-of-words (BOW) features.
### Findings: 
The ensemble model achieved an overall area under the receiving operator characteristics curve (AUROC) of 0.8453 (95% CI: 0.8409-0.8495) significantly outperforming the BOW baseline (0.7548 (95% CI: 0.7493-0.7603)), and an AUROC as high as 0.8691 (95% CI: 0.8658-0.8724) for the high risk category. The transformer-based and graph-based models achieved similar performance to the ensemble, with an AUROC of 0.8363 (95% CI: 0.8318-0.8407) and 0.8395 (95% CI: 0.8352-0.8439), respectively. For the condition and phase strata, Hemic and Lymphatic Diseases (AUROC of 0.884 (95% CI: 0.845-0.899)) and Phase I protocols (AUROC of 0.8493 (95% CI: 0.8408-0.8573)) achieved the highest performance, respectively.
### Interpretation: 
We demonstrate the success of deep learning models in predicting CT risk categories from protocols, paving the way for customized risk mitigation strategies for protocol design and eventual reduction in costs and time-to-market for drugs.

----------------
# Examples: 
In "examples" folder, predicted risk labels and potentially risky individual sections and associated risk scores are presented for the two clinical trials.
* The file "results_NCT00963560.csv" contains the results of the clinical trial with the NCT ID as NCT00963560. The proposed model predicts low risk for various phase-condition combinations, which increases the clinical designer's confidence that the CT protocol is well designed.
* The file "results_NCT01566552.csv" contains the results of the clinical trial with the NCT ID as NCT01566552. The proposed model predicts high risk for various phase-condition combinations. The interpretability module of the proposed model gives a clear insight to the clinical designer as to which section needs improvement for risk mitigation. In case of clinical trial NCT01566552, result shows that Contacts Locations, Design, and Condition sections need to be redesigned to lower the risk.  

