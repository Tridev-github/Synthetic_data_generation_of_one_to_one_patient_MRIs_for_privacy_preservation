# Synthetic_data_generation_of_one_to_one_patient_MRIs_for_privacy_preservation
Abstract—This project introduces a novel approach to address privacy concerns in medical research by employing a Generative Adversarial Network (GAN) for the synthetic generation of
Magnetic Resonance Imaging (MRI) data on a one-to-one patient
basis. The proposed architecture incorporates a modified U-Net as
the generator and a tailored Convolutional Neural Network (CNN) as the discriminator. The U-Net is adapted to capture intricate details in MRI scans, ensuring the generation of realistic and
privacy-preserving synthetic data. The discriminator, with its modifications, effectively distinguishes between synthetic and real
MRI data, validating the authenticity of the generated samples. The training process involves a dynamic interplay between the generator and discriminator, resulting in synthetic MRI data that closely mimics the statistical properties of real patient scans. Evaluation metrics and expert assessments demonstrate the
efficacy of the modified GAN architecture in producing synthetic data that not only safeguards patient privacy but also serves as a
valuable resource for training machine learning models in medical
imaging. This research contributes to advancing privacy- preserving techniques, fostering secure and ethical practices in
healthcare research. Index Terms—Privacy Preservation, Synthetic Data Generation, Generative Adversarial Networks (GANs), Magnetic Resonance
Imaging (MRI), U-Net, Convolutional Neural Network (CNN), Medical Data Privacy, Machine Learning in Healthcare, Image Synthesis. I. INTRODUCTION
The integration of machine learning techniques into medical
research has paved the way for significant advancements in
diagnostic and prognostic capabilities. However, the
utilization of patient-specific data in these models raises
critical concerns regarding privacy and data security. Magnetic Resonance Imaging (MRI), a cornerstone in
medical imaging, contains inherently sensitive patient
information, necessitating the development of robust privacy- preserving methodologies. This paper presents a novel
approach to address this challenge through the
implementation of a Generative Adversarial Network (GAN)
tailored for synthetic MRI data generation on a one-to-one
patient basis. Medical data, particularly MRI scans, play a pivotal role in
training machine learning models for image analysis and
diagnostic tasks. The ethical imperative to preserve patient
privacy while harnessing the power of such data has led to
the exploration of generative models, specifically GANs, for
synthetic data generation. Our work focuses on enhancing the
privacy-preserving capabilities of this process by introducing
modifications to both the generator and discriminator
components of the GAN architecture. The generator, based on a modified U-Net architecture, is
designed to capture intricate details in MRI scans, ensuring
the synthesis of realistic and privacy-preserving data. Simultaneously, the discriminator, a modified Convolutional
Neural Network (CNN), is tailored to effectively distinguish
between synthetic and real MRI data, ensuring the
authenticity of the generated samples. Through a dynamic
interplay between the generator and discriminator during the
training process, our model aims to produce synthetic MRI
data that closely mirrors the statistical properties of real
patient scans.
The significance of this research lies in its contributiontotheadvancement of privacy-preserving techniques in medical datageneration. By facilitating the secure and ethical useofsynthetic MRI data for training machine learning models, ourproposed approach addresses the critical balance betweendatautility and patient confidentiality. The subsequent sectionsdelve into the methodology employed in our modifiedGANarchitecture, presenting experimental results andacomprehensive discussion on the implications andpotentialapplications of our privacy-preserving synthetic MRI datageneration framework. II. RELATED WORK OR LITERATURESURVEYMany researchers have done on comment emotion detection, The paper conducts a survey on synthetic data generation,evaluation methods, and Generative Adversarial Networks(GANs). It aims to provide a comprehensive overviewofthefield and serve as a resource for new researchers. Thereviewincludes querying major databases, analyzing relevant authors,journals, cited papers, research areas, institutions, andGANarchitectures. It also covers common trainingproblems,breakthroughs, and GAN architectures for tabular data. Thepaper discusses algorithms for generating synthetic data, theirapplications, and provides insights. Additionally, it reviewstechniques for evaluating synthetic data quality, especiallyfortabular data. The survey paper aims to fill a gap in the literaturebycombining synthetic data generation and GANs inonecomprehensive review. It serves as a valuable resource for new researchers inthefield,providing an overview of key contributions, references, andinsights into synthetic data generation, GANs, andevaluationmethods. The methodology involves a thorough reviewof existingliterature and databases, offering insights into the most relevantaspects and trends in the field. The paper provides a schematic overviewof the informationpresented, making it a strong starting point for researchersinterested in synthetic data generation and GANs. This paper is uses multiple deep learning approaches likeneuralnetworks and auto encoders to analyze rich EHRdata. Manydifferent applications of EHR were examined under thesetechniques. The authors of this paper have explored the current state-of-the-art applications of deep learning methods in electronichealthrecords. They identify several limitations of recent researchinvolvingtopics such as model interpretability, data heterogeneity, andlack of universal benchmarks. They conclude by summarizing the state of the fieldandidentifying avenues of future deep EHR research. This paperismore of an investigative research that calls out theareasoflacking development in the industry. The areas of applicationthey surveyed were EHR Information Extraction (IE), OutcomePrediction, EHR Representation Learning, ComputationalPhenotyping, and Clinical Data De-Identification. The method involves two learning components: a generatorthatoutputs sanitised public variables (with distortionconstraints)and a discriminator that learns private variables fromthesanitised data. The generator and discriminator competeinaconstrained minimax, zero-sum game to achieve their goals.
The framework is used on 3 image dataset- MNIST, LSUN
and CelebA. Theoretically, they have demonstrated that GANobfuscator
rigorously guarantees differential privacy. In addition, they
have conducted exhaustive experiments to demonstrate that
GANobfuscator can generate high-quality data within
reasonable privacy budgets while retaining its utility. Moreover, their experimental results validate that
GANobfuscator does not suffer from mode collapse or
gradient vanishing during the training procedure, and therefore
can maintain an excellent level of stability and scalability for
model training. This paper makes use of Generative Adversarial Networks
(GANs) to generate synthetic ECG data that retains statistical
and clinical characteristics while anonymizing the data source. This literature survey aims to explore the state of the art in
generating synthetic ECGs using GANs for healthcare data
anonymization. The authors describe an approach for the generation of
synthetic electrocardiograms (ECGs) based on Generative
Adversarial Networks (GANs) with the objective of
anonymizing users’ information for privacy issues.the authors
propose general raw data processing after transformation into
an image, so it can be managed through a GAN, then decoded
back to the original data domain This is intended to create
valuable data that can be used both in educational and research
areas, while avoiding the risk of a sensitive data leakage. The
method follows 5 steps procedure: (i) data standardization, (ii)
data arrangement into an image, (iii) GAN-based architecture
selection, (iv) training concerns, and (v) results evaluation. This paper uses data augmentation techniques that can be used
to create synthetic datasets sufficiently large to train machine
learning models. In this work, we apply the concept of
generative adversarial networks (GANs) to perform a data
augmentation from patient data obtained through IoMT
sensors for Chronic Obstructive Pulmonary Disease (COPD)
monitoring, it also uses AI algorithm to demonstrate the
accuracy of the synthetic data by comparing it to the real data
recorded by the sensors
The authors show the results obtained demonstrate how
synthetic datasets created through a well-structured GAN are
comparable with a real dataset, as validated by a novel
approach based on machine learning. the evaluation of the synthetic dataset generated with the GAN
is performed by using the LLM algorithm compared with the
real information. The results obtained show how the synthetic
dataset is aligned with the real dataset, further demonstrated by
the rules obtained through the LLM algorithm. Adversarial machine learning techniques could be used for the
dataset and algorithms in order to evaluate and validate the
robustness of the platform. Finally, due to the many rules
generated by machine learning algorithms, we will be able to
investigate other statistical validation approaches or automatic
algorithms to assist human decision (e.g., rule distance). III. SYSTEM ARCHITECTURE
The proposed system architecture for privacy-preserving
synthetic MRI data generation is designed around a Cyclic
Generative Adversarial Network (CycleGAN), utilizing a
modified U-Net generator and a modified CNN discriminator. The architecture aims to generate realistic and privacypreserving synthetic MRI data by iteratively translatingreal MRIscans into synthetic counterparts and vice versa. Modified U-Net Generator (unet_generator function):
The U-Net generator is structured with a downsamplingpathandan upsampling path, leveraging a series of convolutional layerswith leaky ReLU activations. The downsampling path(down_stack) consists of successive downsample operations, progressively reducing spatial dimensions while extractinghierarchical features. The upsampling path (up_stack) usesupsample operations to recover spatial information andconcatenate features from the corresponding downsamplinglayer.The generator takes a 4D input tensor of shape [batch_size, height, width, channels] and produces a 4Doutput tensor. It
utilizes skip connections to concatenate features fromthedownsampling and upsampling paths, promoting the preservationof spatial information during synthesis. The final layer usesatransposed convolution to generate synthetic MRI data, employing a hyperbolic tangent activation function. Modified CNN Discriminator (discriminator function):
The CNN discriminator is designed to distinguish betweenreal
and synthetic MRI data. It consists of a series of downsampleoperations, applying leaky ReLU activations and InstanceNormalization to capture hierarchical features. Zero-paddingisused to maintain spatial dimensions, enhancing the
discriminator's ability to capture global context. The final layersinvolve a convolutional operation followed by a linear activationfunction, contributing to the discrimination between real andsynthetic MRI scans. The discriminator takes a 4D input tensor of shape [batch_size, height, width, channels] and produces a 4Doutput tensor. It istrained to distinguish between real and synthetic MRI data, providing feedback to the generator in the adversarial trainingprocess. Cyclic GAN Framework:
The architecture operates within a Cyclic GANframework, facilitating the cyclic translation between real and syntheticMRIdata. The generator and discriminator components are trainedconcurrently using adversarial loss functions. The generator aimsto synthesize MRI data indistinguishable fromreal data, whilethe discriminator strives to correctly classify the originof theinput data. This adversarial training loop iterates, resultinginthegeneration of privacy-preserving synthetic MRI data that alignswith the statistical properties of real patient scans.
Fig. 1. Flow Chart
Fig. 2. PROPOSED Unet MODEL
IV.IMPLEMENTATION
For this project, Stanford Sentiment Treebank Dataset was
The implemented system represents a pioneering approach to
privacy-preserving synthetic Magnetic Resonance Imaging
(MRI) data generation, leveraging a Cyclic Generative
Adversarial Network (CycleGAN) architecture. The core
components of this innovative framework include a modified
U-Net generator (comprising generator_g and generator_f)
and a tailored Convolutional Neural Network (CNN)
discriminator (discriminator_x and discriminator_y). The
following narrative provides a comprehensive insight into the
training step and overarching training loop, illuminating the
intricacies of this research endeavor. In the training step, the generators are entrusted with the task
of executing bidirectional translations, transforming real MRI
scans into synthetic counterparts and vice versa. Thisbidirectional translation is foundational to the cycle-consistentnature of the CycleGAN, reinforcing the network's abilitytomaintain coherence in the synthetic data generationprocess. Tofurther enhance the fidelity of the synthetic data, identitylossmetrics are strategically employed. These metrics ensurethatthe generators refrain from introducing unnecessaryalterationsduring the synthesis, preserving the intrinsic characteristicsofthe input scans. Critical to the success of the adversarial training dynamicsisthe evaluation of discriminator outputs. The discriminators(discriminator_x and discriminator_y) play a pivotal roleinassessing the generators' effectiveness in deceivingtheircounterparts. Adversarial losses, quantified throughmetricssuch as gen_g_loss and gen_f_loss, provide a measureof howwell the generators generate synthetic scans that areindistinguishable from real scans. This adversarial trainingloopserves as a driving force behind the refinement of thegenerative components. The importance of cycle consistency in the translationprocessis underscored by the calculation of the cycle consistencyloss(total_cycle_loss). This metric ensures that the cyclicallyreconstructed scans (cycled_x and cycled_y) alignwiththeoriginal scans, contributing significantly to theoverallcoherence and consistency in the bidirectional translation. Total generator losses (total_gen_g_loss and total_gen_f_loss)serve as comprehensive metrics, combining adversarial losses,cycle consistency losses, and identity losses. Theselossescollectively guide the generators toward the overarchinggoalof generating high-quality, privacy-preserving syntheticMRIdata. Discriminator losses (disc_x_loss and disc_y_loss) areintegralto the training step, quantifying the discriminative capacityofthe network. By assessing the difference betweenreal andsynthetic scans, the discriminators contribute significantlytothe refinement of their capabilities. The broader training loop, unfolding over a predeterminednumber of epochs, involves iterative processing of pairedrealMRI scans from distinct datasets (tr1 and tr2). The trainingstepis invoked for each pair, resulting in a continuous refinement ofthe generative and discriminative components. Periodically, thesystem generates synthetic MRI images using a sampleofthetraining data, offering visual insights into the evolvingnatureof the synthetic data generation process. These generatedimages provide a qualitative perspective on the network'sability to synthesize realistic MRI data. As an integral part of the training process, checkpointsaresystematically saved. These checkpoints serve the dual purposeof facilitating the resumption of training fromspecificpointsand enabling comprehensive model evaluations. Thismeticulous approach to checkpoint management adds alayerofrobustness to the training pipeline, contributingtothereproducibility and reliability of the research outcomes.
V. RESULT AND ANALYSIS
In this project we proposed a system architecture that
enables hospitals to share their image data. The main goal of
this paper was to create a product which generates usable
image data without the need of sharing private patient data. In
the end we were able to observe that when we trained the
model with more training datasets that had more and more
medical images the model lowered loss rate. For this paper we
have collected a dataset of around 200 medical images hence
we observe the generator loss of around 3.5 and discriminator
loss of around 0.5.
Fig. 3. Synthetic MRI
Fig. 4. Metrics for the Model
Fig. 5. HISTORY CURVE FOR LOSS
VI. CONCLUSIONIn summary, this research project introduces a cutting-edgesystem for creating private synthetic Magnetic ResonanceImaging (MRI) data. We used a smart computer setupcalledaCyclic Generative Adversarial Network (CycleGAN) withamodified U-Net generator and a customized CNNdiscriminator.This system excels at translating real MRI scans intosyntheticones and vice versa, maintaining the original details whileavoiding unnecessary changes. The discriminators playakeyrole in training the system to distinguish betweenreal andsynthetic scans. The training process, spread over several cycles,continually improves the network's ability to createprivatesynthetic MRI data. Regularly generating synthetic imagesandmanaging checkpoints enhance the reliability of the project. Overall, this research contributes to the field of medical imagegeneration by presenting a model that can generate syntheticMRI data with a strong focus on patient privacy. The proposedapproach provides valuable insights for using medical datainasecure and ethical manner for research, pushingforwardhealthcare technologies while respecting patient confidentiality.
REFERENCES
[1] PATE-GAN: GENERATING SYNTHETIC DATA WITH
DIFFERENTIAL PRIVACY GUARANTEES - James Jordon, Jinsung
Yoon, Mihaela van der Schaar (2019)
[2] Which Generative Adversarial Network Yields High-Quality Synthetic Medical Images: Investigation Using AMD Image Datasets - Guilherme C. Oliveira , Gustavo H. Rosa, Himesh Kumar (March 2022)
[3] RSurvey on Synthetic Data Generation, Evaluation Methods and GANs
-Alvaro Figueira , Bruno Vaz (2020)
[4] Deep EHR: A Survey of Recent Advances in Deep Learning Techniques
for Electronic Health Record (EHR) Analysis - Benjamin Shickel , Patrick James Tighe , Azra Bihorac , and Parisa Rashidi (2021)
[5] BGANobfuscator: Mitigating Information Leakage Under GAN via Differential Privacy - Chugui Xu, Ju Ren, Deyu Zhang, Yaoxue Zhang, Zhan Qin, and Kui Ren (2021)
[6] Generating Synthetic ECGs Using GANs for Anonymizing Healthcare Data Esteban Piacentino, Alvaro Guarner, and Cecilio Angulo (2022)
[7] A Generative Adversarial Network (GAN) Technique for Internet of
Medical Things Data - Ivan Vaccari,Vanessa Orani, Alessia Paglialonga,Enrico Cambiaso, andMaurizio Mongelli (2019)
[8] Synthetic CT generation from weakly paired MR images using
cycle‐consistent GAN for MR‐guided radiotherapy - Seung Kwan Kang1, Hyun Joon An, Hyeongmin Jin,Jung‐in Kim, Eui Kyu Chie, Jong Min Park, Jae Sung Lee (2023)
[9] Protect and Extend - Using GANs for Synthetic Data Generation of
Time-Series Medical Records Navid Ashrafi1, Vera Schmitt,Robert P. Spang, Sebastian M oller, Jan-NiklasVoigt-Antons (2023)
