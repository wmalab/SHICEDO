echo "Step 1: Filter the cells based on contact number"
python data_filter.py
echo "Step 2: Filter out the inter-chromosomal interactions"
python filter_true_data.py
echo "Step 3: Downsampling the matrix to generate low-resolution input "
python down_sampling_sciHiC.py
echo "Step 4: Run Rscrip to do Bandnorm"
Rscript bandnorm.R
echo "Step 5: Organize normalized result"
python run_bandnorm.py
echo "Step 6: Divide large matrixes into submatrices and save as torch tensor"
python generate_input.py