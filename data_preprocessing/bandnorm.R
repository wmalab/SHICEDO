# change stage1 2 3 4 
library(data.table)
library(Rcpp, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.2/lib64/R/library")
library(SummarizedExperiment, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.2/lib64/R/library")
library(Biobase, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.2/lib64/R/library")
library(MatrixGenerics, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.2/lib64/R/library")
library(ggplot2)
library(Gviz, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.0/lib64/R/library")
# library(GenomicInteractions)
library(MatrixGenerics, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.2/lib64/R/library")
library(BandNorm)

# norm low resolution data
Path <- "process_data/Nagano_process/1mb_con310000/ds_9/stage1"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/ds_9/norm/stage1.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/ds_9/stage2"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/ds_9/norm/stage2.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/ds_9/stage3"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/ds_9/norm/stage3.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/ds_9/stage4"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/ds_9/norm/stage4.txt",sep="\t",row.names=FALSE)

# norm true data
Path <- "process_data/Nagano_process/1mb_con310000/filter_true_no_inter/stage1"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/filter_true_no_inter/filter_true_no_inter_norm/stage1.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/filter_true_no_inter/stage2"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/filter_true_no_inter/filter_true_no_inter_norm/stage2.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/filter_true_no_inter/stage3"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/filter_true_no_inter/filter_true_no_inter_norm/stage3.txt",sep="\t",row.names=FALSE)

Path <- "process_data/Nagano_process/1mb_con310000/filter_true_no_inter/stage4"
bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
write.table(bandnorm_result_test22,"process_data/Nagano_process/1mb_con310000/filter_true_no_inter/filter_true_no_inter_norm/stage4.txt",sep="\t",row.names=FALSE)

# library(Gviz, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.0/lib64/R/library")
# library(BandNorm)

# # library(Gviz, lib.loc = "/bigdata/operations/pkgadmin/opt/linux/centos/8.x/x86_64/pkgs/R/4.2.0/lib64/R/library")
# ##R data.frame in the form of [chr1, binA, binB, count, diag, cell_name]. The column names should be c("chrom", "binA", "binB", "count", "diag", "cell")
# #my_data <- read.table("data/test2.txt")
# #print(my_data)
# #Path <- "data/normed"
# #bandnorm_result_test22 = bandnorm(hic_df = my_data, save = FALSE)
# #print(bandnorm_result_test22)
# ##write.table(bandnorm_result_test22, file = "data/normed/bandnorm_result1.txt",sep = "\t")
# #fwrite(bandnorm_result_test22,"data/normed/bandnorm_result2.txt", sep="\t",col.names = TRUE, row.names = FALSE)
# # nagano list

# Path <- "process_data/Nagano_process/100kb_con310000/ds_3x3/stage1"
# bandnorm_result_test22 = bandnorm(path = Path, save = FALSE)
# write.table(bandnorm_result_test22,"process_data/Nagano_process/100kb_con310000/ds_3x3/norm/stage1.txt",sep="\t",row.names=FALSE)

# ## cool data
# #Path <- "cooldata"
# #bandnorm_result_test22 = bandnorm_cooler(coolerPath = Path,resolution=10000 )
# #print(bandnorm_result_test22)
