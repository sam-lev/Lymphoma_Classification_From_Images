# Lymphoma_Classification_From_Images
NOTE: current code takes a while to compute the pairwise dostance matrix for neighborDistanceDistribution function
      for this reason I have been serializing the data in a seperate file with the cPickle package. Since still
      too large for github it is not contained in the repository but will autogenerate the pickle file on the first run 
      then use a try except catch to later use the file if present or the truncated (to 10000 data points) file can 
      be found as a .zip  here: https://www.cs.utah.edu/~samlev/serializedData.zip
      
      
Methods: 
nucleiMorphology(epsilon, excelFile) 
  -Given an excel fle with nucei perimeter and area computes the radius from those values as assumed circular.
  If circular radii should be nearly equal. If computed radii differ by epsilon labels nuclei as lymphatic and
  mishapen. Nuclei are then clustered by (perimeter, area) and labeled according to neighbors and common labeling 
  assigned.
  
  example call: 
  
  python lymphomaClassification.py -e 0.2 -f './BL results.xlsx' -analysis nucMorph
  
  neighborDistanceDistribution(BLFile, DLFile) 
    -Either uses fitPairwiseDist(pwDist) to calculate a histogram given the pairwise distances between all midpoints 
    computed from the bounding box of the nuclei specified within BLFile and DLFile. 
    - Or uses fitPairwiseDistGMM(pwDistBL, pwDistDL) to generate a lymphoma classifier based on the computed gaussian 
    distribution fit for the pairwise distances between the midpoint of nuclei specified within BLFile and DLFile using 
    a Gaussian Mixed Model approach.
    
    example call: 
    python lymphomaClassification.py -analysis nDistDist -fDL '../data/DLBCL results.xlsx' -fBL '../data/BL results.xlsx'

