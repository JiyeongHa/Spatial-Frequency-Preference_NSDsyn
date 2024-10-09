# Spatial Frequency Maps in Human Visual Cortex:A Replication and Extension


This repository includes codes for spatial frequency preferences in human  visual cortex (*manuscript in prep*).  

## Abstract
**[Ha, Broderick, Kay, & Winawer (2022; JoV)](https://jov.arvojournals.org/article.aspx?articleid=2792643)** </br>
Neurons in primary visual cortex (V1) of non-human primates are tuned to spatial frequency, with preferred frequency declining with eccentricity. fMRI studies show that spatial frequency tuning can be measured at the mm scale in humans (single voxels), and confirm that preferred frequency declines with eccentricity. Recently, fMRI-based quantitative models of spatial frequency have been developed, both at the scale of voxels (Aghajari, Vinke, & Ling, 2020, J Neurophys) and maps (Broderick, Simoncelli, & Winawer, 2022, JoV). For the voxel-level approach, independent spatial frequency tuning curves were fit to each voxel. For the map-level approach, a low dimensional parameterization (9 parameters) described spatial frequency tuning across all of V1 as a function of voxel eccentricity, voxel polar angle, and stimulus orientation. Here, we sought to replicate and extend Broderick et al.’s results using an independent dataset (Natural scenes dataset, NSD; Allen et al, 2022, Nat Neurosci). Despite many experimental differences between Broderick et al and NSD, including field strength (3T vs 7T), number of stimulus presentations per observer (96 vs 32), and stimulus field of view (12° vs 4.2° maximal eccentricity), most, though not all, of the model parameters showed good agreement. Notably, parameters that capture the dependency of preferred spatial frequency on voxel eccentricity, cardinal vs oblique stimulus orientation, and radial vs tangential stimulus orientation, were similar. We also extended Broderick et al.’s results by fitting the same parametric model to NSD data from V2 and V3. From V1 to V2 to V3, there was an increasingly sharp decline in preferred spatial frequency as a function of eccentricity, and an increasingly large bandwidth in the voxel spatial frequency tuning functions. Together, the results show robust reproducibility of visual fMRI experiments, and bring us closer to a systematic characterization of spatial encoding in the human visual system.


![tuning](https://github.com/JiyeongHa/spatial_frequency/blob/master/example/sf-fig3.png)  
Fig 1. Example spatial frequency tuning curves for NSD. Each panel shows data points binned by eccentricity fit by log Gaussian tuning curves in an example NSD subject.

![Experimental design](https://github.com/JiyeongHa/spatial_frequency/blob/master/example/sf-expdesign.png)  
Fig 2. Differences in experimental details between Broderick et al. (2022) and NSD synthetic 

![Main result](https://github.com/JiyeongHa/spatial_frequency/blob/master/example/sf-results.png)  
Fig 3. Good agreement with tow of the recent studies 