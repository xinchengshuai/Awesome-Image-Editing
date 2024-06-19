# Awesome Image Editing 
This is the github repository of our work "A Survey of Multimodal-Guided Image Editing with Text-to-Image Diffusion Models". 
<!-- We categorize the reviewed papers by their editing scenario, and illustrate their inversion and editing algorithms. -->


## Editing Tasks Discussed in Our Survey
<p align="center">
  <img src="./images/editing_task.jpg" alt="image" style="width:800px;">
</p>

## Unified Framework
<p align="center">
  <img src="./images/unified_framework.jpg" alt="image" style="width:800px;">
</p>



## Table of contents
Content-Aware Editing
- [Object Manipulation + Attribute Manipulation](#object-manipulation-and-attribute-manipulation)
- [Attribute Manipulation](#attribute-manipulation)
- [Spatial Transformation](#spatial-transformation)
- [Inpainting](#inpainting)
- [Style Change](#style-change)
- [Image Translation](#image-translation)

<br>

Content-Free Editing
- [Subject-Driven Customization](#subject-driven-customization)
- [Attribute-Driven Customization](#attribute-driven-customization)

<br>

Experiment and Data
- [Data](#data)



<br>

## Object Manipulation and Attribute Manipulation:
### 1. Training-Free Approaches 
  [📄 UniTune: Text-Driven Image Editing by Fine Tuning a Diffusion Model on a Single Image](https://arxiv.org/abs/2210.09477) | 📖 TOG 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 
   
  [📄 Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion](https://arxiv.org/abs/2303.08767) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$| [🌐 Code](https://github.com/HiPer0/HiPer) 

  [📄 Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{edit}^{Blend}$ | [🌐 Code] 

  [📄 Forgedit: Text Guided Image Editing via Learning and Forgetting](https://arxiv.org/abs/2309.10556) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Blend}$ | [🌐 Code](https://github.com/witcherofresearch/Forgedit) 
   
  [📄 Doubly Abductive Counterfactual Inference for Text-based Image Editing](https://arxiv.org/abs/2403.02981) | 📖 CVPR 2024 | 🔀 $F_{inv}^T+F_{edit}^{Blend}$ | [🌐 Code](https://github.com/xuesong39/DAC) 

  [📄 SINE: Sinle Image Editing with Text-to-Image Diffusion Models](https://arxiv.org/abs/2212.04489) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{edit}^{Score}$ | [🌐 Code](https://github.com/zhang-zx/SINE) 


  [📄 EDICT: Exact Diffusion Inversion via Coupled Transformations](https://arxiv.org/abs/2211.12446) | 📖 CVPR 2023  | 🔀 $F_{inv}^F+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/salesforce/EDICT) 

  [📄 Exact Diffusion Inversion via Bi-directional Integration Approximation](https://arxiv.org/abs/2307.10829) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Norm}$ | [🌐 Code]() 

  [📄 Effective Real Image Editing with Accelerated Iterative Diffusion Inversion](https://arxiv.org/abs/2309.04907) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Norm}$ | [🌐 Code]() 



  [📄 Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/abs/2211.09794) | 📖 CVPR 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images) 

  [📄 Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models](https://arxiv.org/abs/2305.16807) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄ProxEdit: Improving Tuning-Free Real Image Editing with Proximal Guidance](https://arxiv.org/pdf/2306.05414) | 📖 WACV 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/phymhan/prompt-to-prompt) 
   
  [📄 Fixed-point Inversion for Text-to-image diffusion models](https://arxiv.org/abs/2312.12540v1) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code](https://arxiv.org/abs/2310.01506) | 📖 ICLR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$| [🌐 Code](https://github.com/cure-lab/PnPInversion) 

  [📄 Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing](https://arxiv.org/abs/2309.15664) | 📖 NeurIPS 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄 An Edit Friendly DDPM Noise Space: Inversion and Manipulations](https://arxiv.org/abs/2304.06140) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/inbarhub/DDPM_inversion) 

  [📄 Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) | 📖 ICLR 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images) 

  [📄 Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation](https://arxiv.org/abs/2211.12572) | 📖 CVPR 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/MichalGeyer/plug-and-play) 

  [📄 Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing](https://arxiv.org/abs/2403.03431) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing) 

  [📄 StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing](https://arxiv.org/abs/2303.15649) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/sen-mao/StyleDiffusion) 


  
  [📄 Prompt Tuning Inversion for Text-Driven Image Editing Using Diffusion Models](https://arxiv.org/abs/2305.04441) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code]() 

  [📄 Object-aware Inversion and Reassembly for Image Editing](https://arxiv.org/abs/2310.12149) | 📖 ICLR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code](https://github.com/aim-uofa/OIR) 

  [📄 DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427) | 📖 ICLR 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code]() 

  [📄 PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing](https://arxiv.org/abs/2306.16894) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code]() 

  [📄 Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models](https://arxiv.org/abs/2212.08698) | 📖 CVPR 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$| [🌐 Code](https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement) 


  [📄 Noise Map Guidance: Inversion with Spatial Context for Real Image Editing](https://arxiv.org/abs/2402.04625) | 📖 ICLR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://github.com/hansam95/NMG) 
  
  [📄 pix2pix-zero](https://arxiv.org/abs/2302.03027) | 📖 SIGGRAPH 2023 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://github.com/pix2pixzero/pix2pix-zero) 

  [📄 SEGA: Instructing Diffusion using Semantic Dimensions](https://export.arxiv.org/abs/2301.12247v1) | 📖 NeurIPS 2023 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code]() 

  [📄 The Stable Artist: Steering Semantics in Diffusion Latent Space](https://arxiv.org/abs/2212.06013) | 📖 Arxiv 2022 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code]() 

  [📄 LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance](https://arxiv.org/abs/2307.00522) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://huggingface.co/spaces/editing-images/ledits/tree/main) 

  [📄 LEDITS++: Limitless Image Editing using Text-to-Image Models](https://arxiv.org/abs/2311.16711) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp) 

  [📄 Magicremover: Tuning-free Text-guided Image inpainting with Diffusion Models](https://arxiv.org/abs/2310.02848) | 📖 ICLR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code]() 


  [📄 Region-Aware Diffusion for Zero-shot Text-driven Image Editing](https://arxiv.org/abs/2302.11797) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code]() 

  [📄 Delta Denoising Score](https://arxiv.org/abs/2304.07090) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code](https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb) 

  [📄 Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://arxiv.org/abs/2311.18608) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code](https://github.com/HyelinNAM/ContrastiveDenoisingScore) 

  [📄 Ground-A-Score: Scaling Up the Score Distillation for Multi-Attribute Editing](https://arxiv.org/abs/2403.13551) | 📖 Arxiv 2024 | 🔀 $F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code](https://github.com/Ground-A-Score/Ground-A-Score/) 



  [📄 Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models](https://arxiv.org/abs/2305.15779) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄 Photoswap: Personalized Subject Swapping in Images](https://arxiv.org/abs/2305.18286) | 📖 NeurIPS 2023 | 🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/eric-ai-lab/photoswap) 

  [📄 DreamEdit: Subject-driven Image Editing](https://arxiv.org/abs/2306.12624) | 📖 TMLR 2023 |🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code](https://github.com/DreamEditBenchTeam/DreamEdit) 




### 2. Training-Based Approaches


  [📄 InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800) | 📖 CVPR 2023 | [🌐 Code](https://github.com/timothybrooks/instruct-pix2pix) 

  [📄 MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing](https://arxiv.org/abs/2306.10012) | 📖 NeurIPS 2023 | [🌐 Code](https://github.com/OSU-NLP-Group/MagicBrush) 

  [📄 HIVE: Harnessing Human Feedback for Instructional Visual Editing](https://arxiv.org/abs/2303.09618) | 📖 Arxiv 2023 | [🌐 Code](https://github.com/salesforce/HIVE) 


  [📄 Emu Edit: Precise Image Editing via Recognition and Generation Tasks](https://arxiv.org/abs/2311.10089) | 📖 Arxiv 2023 |  [🌐 Code](https://emu-edit.metademolab.com/) 

  [📄 GUIDING INSTRUCTION-BASED IMAGE EDITING VIA MULTIMODAL LARGE LANGUAGE MODELS](https://arxiv.org/abs/2309.17102) | 📖 ICLR 2024 | [🌐 Code](https://mllm-ie.github.io/) 

  [📄 SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models](https://arxiv.org/abs/2312.06739) | 📖CVPR 2024 | [🌐 Code](https://github.com/TencentARC/SmartEdit) 

  [📄 Referring Image Editing: Object-level Image Editing via Referring Expressions](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Referring_Image_Editing_Object-level_Image_Editing_via_Referring_Expressions_CVPR_2024_paper.html) | 📖CVPR 2024  | [🌐 Code]() 


<br>

## Attribute Manipulation:
### 1. Training-Free Approaches

  [📄 KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing](https://arxiv.org/abs/2309.16608) | 📖 PRCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄 Localizing Object-level Shape Variations with Text-to-Image Diffusion Models](https://arxiv.org/abs/2303.11306) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/orpatashnik/local-prompt-mixing) 

  [📄 MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://arxiv.org/abs/2304.08465) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/TencentARC/MasaCtrl) 
   
  [📄 Tuning-Free Inversion-Enhanced Control for Consistent Image Editing](https://arxiv.org/abs/2312.14611) | 📖 AAAI 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 

  [📄 Cross-Image Attention for Zero-Shot Appearance Transfer](https://arxiv.org/abs/2311.03335) | 📖 SIGGRAPH 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/garibida/cross-image-attention) 



### 2. Training-Based Approaches

<br>


## Spatial Transformation:
### 1. Training-Free Approaches

  [📄 DesignEdit: Multi-Layered Latent Decomposition and Fusion for Unified & Accurate Image Editing](https://arxiv.org/abs/2403.14487) | 📖 Arxiv 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/design-edit/DesignEdit) 
 
  [📄 Diffusion Self-Guidance for Controllable Image Generation](https://arxiv.org/abs/2306.00986) | 📖 NeurIPS 2023 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://dave.ml/selfguidance/) 

  [📄 DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models](https://arxiv.org/abs/2307.02421) | 📖 ICLR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://github.com/MC-E/DragonDiffusion) 

 


  [📄 DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing](https://arxiv.org/abs/2306.14435) | 📖 ICLR 2024 | 🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code](https://github.com/Yujun-Shi/DragDiffusion) 

  [📄 DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing](https://arxiv.org/abs/2402.02583) | 📖 ICLR 2024 | 🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Score}$ | [🌐 Code](https://github.com/MC-E/DragonDiffusion) 


### 2. Training-Based Approaches




<br>

## Inpainting:
### 1. Training-Free Approaches
  [📄 HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models](https://arxiv.org/abs/2312.14091) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$  | [🌐 Code](https://github.com/Picsart-AI-Research/HD-Painter) 

  [📄 TF-ICON: Diffusion-Based Training-Free Cross-Domain Image Composition](https://arxiv.org/abs/2307.12493) | 📖 ICCV 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$ | [🌐 Code](https://github.com/Shilin-LU/TF-ICON) 

  [📄 Blended Latent Diffusion](https://arxiv.org/abs/2206.02779) | 📖 TOG 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$  | [🌐 Code](https://github.com/omriav/blended-latent-diffusion) 

  [📄 High-Resolution Image Editing via Multi-Stage Blended Diffusion](https://arxiv.org/abs/2210.12965) | 📖 Arxiv 2022 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$  | [🌐 Code](https://github.com/pfnet-research/multi-stage-blended-diffusion) 

  [📄 Differential Diffusion: Giving Each Pixel Its Strength](https://arxiv.org/abs/2306.00950) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$  | [🌐 Code](https://github.com/exx8/differential-diffusion) 
   
  [📄 Tuning-Free Image Customization with Image and Text Guidance](https://arxiv.org/abs/2403.12658) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Blend}$  | [🌐 Code]() 


  <!-- [📄 DreamEdit]() | [📖 ] | [Inversion+Editing] | [🌐 Code]()  -->




### 2. Training-Based Approaches


  [📄 Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting](https://arxiv.org/pdf/2212.06909) | 📖 CVPR 2024| [🌐 Code](https://imagen.research.google/editor/) 

  [📄 SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model](https://arxiv.org/abs/2212.05034) | 📖 CVPR 2023 | [🌐 Code]() 


  [📄 A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting](https://arxiv.org/abs/2312.03594) | 📖 Arxiv 2023 | [🌐 Code](https://github.com/open-mmlab/PowerPaint) 

  [📄 Paint by Example: Exemplar-based Image Editing with Diffusion Models](https://arxiv.org/abs/2211.13227) | 📖 CVPR 2023 | [🌐 Code](https://github.com/Fantasy-Studio/Paint-by-Example) 

  [📄 ObjectStitch: Object Compositing with Diffusion Model](https://arxiv.org/abs/2212.00932) | 📖 CVPR 2023 | [🌐 Code]() 
   
  [📄 Reference-based Image Composition with Sketch via Structure-aware Diffusion Model](https://arxiv.org/abs/2304.09748) | 📖 CVPR 2023  | [🌐 Code]() 


  [📄 Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model](https://arxiv.org/abs/2306.07596) | 📖 ICASSP 2024 | [🌐 Code](https://sites.google.com/view/phd-demo-page) 

  [📄 AnyDoor: Zero-shot Object-level Image Customization](https://arxiv.org/abs/2307.09481) | 📖 CVPR 2024 | [🌐 Code](https://github.com/ali-vilab/AnyDoor) 


<br>


## Style Change:
### 1. Training-Free Approaches
  [📄 Inversion-Based Style Transfer with Diffusion Models](https://arxiv.org/abs/2211.13203) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{inv}^F+F_{edit}^{Attn}$  | [🌐 Code](https://github.com/zyxElsa/InST) 

  [📄 Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer](https://arxiv.org/abs/2312.09008) | 📖 CVPR 2024 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$  | [🌐 Code](https://github.com/jiwoogit/StyleID) 

  [📄 Z∗: Zero-shot Style Transfer via Attention Rearrangement](https://arxiv.org/abs/2311.16491) | 📖 Arxiv 2023 | 🔀 $F_{inv}^F+F_{edit}^{Attn}$ | [🌐 Code]() 


### 2. Training-Based Approaches

<br>


## Image Translation:
### 1. Training-Free Approaches
  [📄 FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition](https://arxiv.org/abs/2312.07536) | 📖 CVPR 2024 | [🌐 Code](https://github.com/genforce/freecontrol) 

### 2. Training-Based Approaches
  [📄 Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) | 📖 ICCV 2023 | [🌐 Code]() 

  [📄 T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) | 📖 AAAI 2024 | [🌐 Code](https://github.com/TencentARC/T2I-Adapter) 
   
  [📄 SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392) | 📖 CVPR 2024 | [🌐 Code](https://github.com/ali-vilab/SCEdit) 


  [📄 Cocktail: Mixing Multi-Modality Controls for Text-Conditional Image Generation](https://arxiv.org/abs/2306.00964) | 📖 NeurIPS 2023 | [🌐 Code](https://github.com/mhh0318/Cocktail) 

  [📄 Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Model](https://arxiv.org/abs/2305.16322) | 📖 NeurIPS 2023 | [🌐 Code](https://github.com/ShihaoZhaoZSH/Uni-ControlNet) 

  [📄 CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation](https://arxiv.org/abs/2310.13165) | 📖 NeurIPS 2023 | [🌐 Code](https://github.com/sled-group/CycleNet) 

  [📄 One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036) | 📖 Arxiv 2024] | [🌐 Code](https://github.com/GaParmar/img2img-turbo) 

<br>

## Subject-Driven Customization:
### 1. Training-Free Approaches
  [📄 An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) | 📖 ICLR 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/rinongal/textual_inversion) 
   
  [📄 DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Positive-Negative Prompt-Tuning](https://arxiv.org/abs/2211.11337) | 📖 Arxiv 2022 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 

  <!-- [📄 Cones2]() | [📖 ] | [Inversion+Editing] | [🌐 Code]()  -->
   
  [📄 P+: Extended Textual Conditioning in Text-to-Image Generation](https://arxiv.org/abs/2303.09522) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://prompt-plus.github.io/) 

  [📄 A Neural Space-Time Representation for Text-to-Image Personalization](https://arxiv.org/abs/2305.15391) | 📖 TOG 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/NeuralTextualInversion/NeTI) 
   
  [📄 DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://dreambooth.github.io/) 


  [📄A Data Perspective on Enhanced Identity Preservation for Diffusion Personalization](https://arxiv.org/abs/2311.04315) | 📖 ICLR 2024 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 
   
  [📄 FaceChain-SuDe: Building Derived Class to Inherit Category Attributes for One-shot Subject-Driven Generation](https://arxiv.org/abs/2403.06775) | 📖 CVPR 2024 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/modelscope/facechain) 

  [📄 Multi-Concept Customization of Text-to-Image Diffusion](https://arxiv.org/abs/2212.04488) | 📖 CVPR 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/adobe-research/custom-diffusion) 

  [📄 Cones: Concept Neurons in Diffusion Models for Customized Generation](https://arxiv.org/abs/2303.05125) | 📖 ICML 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 
   
  [📄 SVDiff: Compact Parameter Space for Diffusion Fine-Tuning](https://arxiv.org/abs/2303.11305) | 📖 ICCV 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/mkshing/svdiff-pytorch) 


  [📄 Low-Rank Adaptation for Fast Text-to-Image Diffusion Fine-Tuning]() | 📖 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$| [🌐 Code](https://github.com/cloneofsimo/lora) 

  [📄 A Closer Look at Parameter-Efficient Tuning in Diffusion Models](https://arxiv.org/abs/2303.18181) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/Xiang-cd/unet-finetune) 
   
  <!-- [📄 CatVersion]() | [📖 ] | [Inversion+Editing] | [🌐 Code]()  -->


  [📄 Break-a-scene: Extracting multiple concepts from a single image](https://arxiv.org/abs/2305.16311) | 📖 SIGGRAPH 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/google/break-a-scene) 

  [📄 Clic: Concept Learning in Context](https://arxiv.org/abs/2311.17083) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://mehdi0xc.github.io/clic/) 
   
  [📄 Disenbooth: Disentangled parameter-efficient tuning for subject-driven text-to-image generation](https://arxiv.org/abs/2305.03374) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/forchchch/DisenBooth) 
  
  [📄 Decoupled Textual Embeddings for Customized Image Generation](https://arxiv.org/abs/2312.11826) | 📖 AAAI 2024 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/PrototypeNx/DETEX) 

  [📄 ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://arxiv.org/abs/2306.00971) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Attn}$ | [🌐 Code](https://github.com/haoosz/ViCo) 
   
  [📄 DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization](https://arxiv.org/abs/2402.09812) | 📖 CVPR 2024 | 🔀 $F_{inv}^T+F_{edit}^{Attn}$ | [🌐 Code](https://ku-cvlab.github.io/DreamMatcher/) 


  [📄 Pick-and-Draw: Training-free Semantic Guidance for Text-to-Image Personalization](https://arxiv.org/abs/2401.16762) | 📖 Arxiv 2024 | 🔀 $F_{inv}^F+F_{edit}^{Optim}$ | [🌐 Code]() 
  

### 2. Training-Based Approaches

  [📄 Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models](https://arxiv.org/abs/2304.02642) | 📖 ICLR 2024 | [🌐 Code]() 

  [📄 InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning](https://arxiv.org/abs/2304.03411) | 📖 CVPR 2024] | [🌐 Code](https://jshi31.github.io/InstantBooth/) 
   
  [📄 Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models](https://arxiv.org/abs/2302.12228) | 📖 Arxiv 2023 | [🌐 Code](https://tuning-encoder.github.io/) 
  
  [📄 Enhancing Detail Preservation for Customized Text-to-Image Generation: A Regularization-Free Approach](https://arxiv.org/abs/2305.13579) | 📖 ICLR 2024 | [🌐 Code](https://github.com/drboog/ProFusion) 

  [📄 FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://arxiv.org/abs/2305.10431) | 📖 Arxiv 2023 | [🌐 Code]() 
   
  [📄 PhotoMaker: Customizing Realistic Human Photos via Stacked {ID} Embedding](https://arxiv.org/abs/2312.04461) | 📖 Arxiv 2023 |  [🌐 Code](https://github.com/TencentARC/PhotoMaker) 

  [📄 PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models](https://arxiv.org/abs/2309.05793) | 📖 Arxiv 2023 | [🌐 Code](https://photoverse2d.github.io/) 


  [📄 InstantID: Zero-shot Identity-Preserving Generation in Seconds](https://arxiv.org/abs/2401.07519) | 📖 Arxiv 2024 | [🌐 Code](https://github.com/InstantID/InstantID) 

  [📄 ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation](https://arxiv.org/abs/2302.13848) | 📖 ICCV 2023 | [🌐 Code](https://github.com/csyxwei/ELITE) 
   
  [📄 BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing](https://arxiv.org/abs/2305.14720) | 📖 NeurIPS 2023 | [🌐 Code]() 
  
  [📄 Domain-Agnostic Tuning-Encoder for Fast Personalization of Text-To-Image Models](https://arxiv.org/abs/2307.06925) | 📖 SIGGRAPH 2023 | [🌐 Code](https://arxiv.org/abs/2305.14720) 

  [📄 Unified Multi-Modal Latent Diffusion for Joint Subject and Text Conditional Image Generation](https://arxiv.org/abs/2303.09319) | 📖 Arxiv 2023 | [🌐 Code]() 
   
  [📄 Subject-Diffusion: Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning](https://arxiv.org/abs/2307.11410) | 📖 Arxiv 2023 | [🌐 Code]() 

  [📄 Instruct-Imagen: Image Generation with Multi-modal Instruction](https://arxiv.org/abs/2401.01952) | 📖 Arxiv 2024 | [🌐 Code]() 



<br>


## Attribute-Driven Customization:
### 1. Training-Free Approaches

  [📄 ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models](https://arxiv.org/abs/2305.16225) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/zyxElsa/ProSpect) 

  [📄 An Image is Worth Multiple Words: Multi-attribute Inversion for Constrained Text-to-Image Synthesis](https://arxiv.org/abs/2311.11919) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 
   
  [📄 Concept Decomposition for Visual Exploration and Inspiration](https://arxiv.org/abs/2305.18203) | 📖 TOG 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/google/inspiration_tree) 
  
  [📄 ReVersion: Diffusion-Based Relation Inversion from Images](https://arxiv.org/abs/2303.13495) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://github.com/ziqihuangg/ReVersion) 

  [📄 Learning Disentangled Identifiers for Action-Customized Text-to-Image Generation](https://arxiv.org/abs/2311.15841) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://adi-t2i.github.io/ADI/) 
   
  [📄 Lego: Learning to Disentangle and Invert Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](https://arxiv.org/abs/2311.13833) | 📖 Arxiv 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code]() 

  [📄 StyleDrop: Text-to-Image Generation in Any Style](https://arxiv.org/abs/2306.00983) | 📖 NeurIPS 2023 | 🔀 $F_{inv}^T+F_{edit}^{Norm}$ | [🌐 Code](https://styledrop.github.io/) 






### 2. Training-Based Approaches
  [📄 ArtAdapter: Text-to-Image Style Transfer using Multi-Level Style Encoder and Explicit Adaptation](https://arxiv.org/abs/2312.02109) | 📖 Arxiv 2023 | [🌐 Code](https://github.com/cardinalblue/ArtAdapter) 

  [📄 DreamCreature: Crafting Photorealistic Virtual Creatures from Imagination](https://arxiv.org/abs/2311.15477) | 📖 Arxiv 2023 | [🌐 Code]() 
   
  [📄 Language-Informed Visual Concept Learning](https://arxiv.org/abs/2312.03587) | 📖 ICLR 2024| [🌐 Code](https://cs.stanford.edu/~yzzhang/projects/concept-axes/) 

  [📄 pOps: Photo-Inspired Diffusion Operators](https://arxiv.org/abs/2406.01300) | 📖 Arxiv 2024 | [🌐 Code](https://github.com/pOpsPaper/pOps) 