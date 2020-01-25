install.packages(c("RPostgreSQL"))
install.packages("RcppNumerical")

library(RPostgreSQL)
library("RcppNumerical")
library(ggplot2)
library(plyr)
library(fastLR)
drv <- dbDriver('PostgreSQL')
db <- dbConnect(drv, host='127.0.0.1', user='postgres', dbname='postgres', password='postgres', port='5432')

# herohe_data = (dbGetQuery(db, "SELECT ip.id, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
#   FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE nucleus_circularity > 0.4
#                          AND nucleus_area <= 100;"))
# circ_file = dbGetQuery(db, "SELECT 
# MAX(her2status), filename, nucleus_circularity, COUNT(*), COUNT(*) / 20008012., log(COUNT(*) / 20008012.)
# FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
#  WHERE her2status = 'Negative' GROUP BY nucleus_circularity, filename
# UNION 
# SELECT MAX(her2status), filename, nucleus_circularity, COUNT(*), COUNT(*) / 20098881., log(COUNT(*) / 20098881.,
# )
#   FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
# WHERE her2status = 'Positive' GROUP BY nucleus_circularity, filename;")
# 
# area_file = dbGetQuery(db, "SELECT 
# MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20008012., log(COUNT(*) / 20008012.)
# FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
#  WHERE her2status = 'Negative' GROUP BY nucleus_area, filename
# UNION 
# SELECT MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20098881., log(COUNT(*) / 20098881.)
#   FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
# WHERE her2status = 'Positive' GROUP BY nucleus_area, filename;")
# 
# perimeter_file = dbGetQuery(db, "SELECT 
# MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20008012., log(COUNT(*) / 20008012.)
# FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
#  WHERE her2status = 'Negative' GROUP BY nucleus_area, filename
# UNION 
# SELECT MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20098881., log(COUNT(*) / 20098881.)
#   FROM public.images i 
# JOIN public.image_parameter ip ON i.id = ip.image_id
# WHERE her2status = 'Positive' GROUP BY nucleus_area, filename;")
# 
# 
# test = dbGetQuery(db, "SELECT
# MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20008012., log(COUNT(*) / 20008012.)
# FROM public.images i
# JOIN public.image_parameter ip ON i.id = ip.image_id
#  WHERE her2status = 'Negative' GROUP BY nucleus_area, filename
# UNION
# SELECT MAX(her2status), filename, nucleus_area, COUNT(*), COUNT(*) / 20098881., log(COUNT(*) / 20098881.)
#   FROM public.images i
# JOIN public.image_parameter ip ON i.id = ip.image_id
# WHERE her2status = 'Positive' GROUP BY nucleus_area, filename;")

herohe_data_xl_neg = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE (nucleus_circularity > 0.3
                         AND nucleus_area > 75)  AND her2Status = 'Negative';"))

herohe_data_l_neg = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE ((nucleus_circularity > 0.3
                         AND nucleus_area <= 75) AND nucleus_area > 45)  AND her2Status = 'Negative';"))

herohe_data_m_neg = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE ((nucleus_circularity > 0.3
                         AND nucleus_area <= 45) AND nucleus_area > 25)  AND her2Status = 'Negative';"))

herohe_data_s_neg = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE nucleus_circularity > 0.3
                         AND nucleus_area < 25  AND her2Status = 'Negative';"))

herohe_data_xl_pos = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE (nucleus_circularity > 0.3
                         AND nucleus_area > 75)  AND her2Status = 'Positive';"))

herohe_data_l_pos = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE ((nucleus_circularity > 0.3
                         AND nucleus_area <= 75) AND nucleus_area > 45)  AND her2Status = 'Positive';"))

herohe_data_m_pos = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE ((nucleus_circularity > 0.3
                         AND nucleus_area <= 45) AND nucleus_area > 25)  AND her2Status = 'Positive';"))

herohe_data_s_pos = (dbGetQuery(db, "SELECT ip.id, filename, her2status, nucleus_area, nucleus_perimeter, nucleus_circularity
  FROM public.images i JOIN public.image_parameter ip ON i.id = ip.image_id WHERE nucleus_circularity > 0.3
                         AND nucleus_area < 25  AND her2Status = 'Positive';"))

test1 = sort(unique(herohe_data_xl_neg$filename))
test2 = sort(unique(herohe_data_l_neg$filename))
test3 = sort(unique(herohe_data_m_neg$filename))
test4 = sort(unique(herohe_data_s_neg$filename))

test5 = sort(unique(herohe_data_xl_pos$filename))
test6 = sort(unique(herohe_data_l_pos$filename))
test7 = sort(unique(herohe_data_m_pos$filename))
test8 = sort(unique(herohe_data_s_pos$filename))
tests_neg = cbind(test1,test2,test3,test4)
tests_pos = cbind(test5,test6,test7,test8)
her_pos = rbind(herohe_data_xl_pos, herohe_data_l_pos, herohe_data_m_pos, herohe_data_s_pos)
her_neg = rbind(herohe_data_xl_neg, herohe_data_l_neg, herohe_data_m_neg, herohe_data_s_neg)

herohe_data_l_neg$nucleus_circularity[herohe_data_l_neg$filename == "367.mrxs"]
test = her_neg[her_neg$filename == "367.mrxs",]$nucleus_area < 85 
range(test$nucleus_area)

slide_mat_neg = c()
class_vec_neg = c()
slide_names_neg = c()
slide_names_pos = c()
# read positive
## one Value missing for XL on one slide all the others are equivalent use test1 to test8 to check 
## read in dummy slide vector get area circulatrity and basically whatever column there is is yours 
## reads for the four tresholded groups above the structure is [filename, X1=area_, X2]
for (i in 1:length(tests[,2])){
  slide_vals = c()
  class_vec_neg = cbind(class_vec_neg, c(0))
  slide_names_neg = rbind(slide_names_neg, tests_neg[,2])
  
  slide_vals = cbind(slide_vals, length(herohe_data_xl_neg[herohe_data_xl_neg$filename == tests_neg[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_l_neg[herohe_data_l_neg$filename == tests_neg[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_m_neg[herohe_data_m_neg$filename == tests_neg[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_s_neg[herohe_data_s_neg$filename == tests_neg[, 2][i],]$nucleus_area))
  slide_vals = slide_vals/sum(slide_vals)
  slide_names_neg = cbind(slide_names_neg[,2][i], slide_names_neg)

  slide_vals = cbind(slide_vals, mean(herohe_data_xl_neg[herohe_data_xl_neg$filename == tests_neg[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_l_neg[herohe_data_l_neg$filename == tests_neg[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_m_neg[herohe_data_m_neg$filename == tests_neg[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_s_neg[herohe_data_s_neg$filename == tests_neg[, 2][i],]$nucleus_circularity))
  
  slide_vals = cbind(slide_vals, mean(herohe_data_xl_neg[herohe_data_xl_neg$filename == tests_neg[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_l_neg[herohe_data_l_neg$filename == tests_neg[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_m_neg[herohe_data_m_neg$filename == tests_neg[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_s_neg[herohe_data_s_neg$filename == tests_neg[, 2][i],]$nucleus_perimeter))

  slide_mat_neg = rbind(slide_mat_neg, slide_vals)
  print(i)
  print(length(tests_neg[, 2]))
}
slide_mat_pos = c()
class_vec_pos = c()
for(i in 1:length(tests_neg[, 2])){
  slide_vals = c()
  class_vec_pos = cbind(class_vec_pos, c(1))
  slide_names_pos = rbind(slide_names_pos, tests_pos[,2])
  
  slide_vals = cbind(slide_vals, length(herohe_data_xl_pos[herohe_data_xl_pos$filename == tests_pos[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_l_pos[herohe_data_l_pos$filename == tests_pos[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_m_pos[herohe_data_m_pos$filename == tests_pos[, 2][i],]$nucleus_area))
  slide_vals = cbind(slide_vals, length(herohe_data_s_pos[herohe_data_s_pos$filename == tests_pos[, 2][i],]$nucleus_area))

  slide_vals = slide_vals/sum(slide_vals)
  slide_names_pos = cbind(slide_names_pos[,2][i], slide_names_pos)
  
  slide_vals = cbind(slide_vals, mean(herohe_data_xl_pos[herohe_data_xl_pos$filename == tests_pos[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_l_pos[herohe_data_l_pos$filename == tests_pos[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_m_pos[herohe_data_m_pos$filename == tests_pos[, 2][i],]$nucleus_circularity))
  slide_vals = cbind(slide_vals, mean(herohe_data_s_pos[herohe_data_s_pos$filename == tests_pos[, 2][i],]$nucleus_circularity))

  
  slide_vals = cbind(slide_vals, mean(herohe_data_xl_pos[herohe_data_xl_pos$filename == tests_pos[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_l_pos[herohe_data_l_pos$filename == tests_pos[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_m_pos[herohe_data_m_pos$filename == tests_pos[, 2][i],]$nucleus_perimeter))
  slide_vals = cbind(slide_vals, mean(herohe_data_s_pos[herohe_data_s_pos$filename == tests_pos[, 2][i],]$nucleus_perimeter))

  slide_mat_pos = rbind(slide_mat_pos, slide_vals)
  print(i)
  print(length(tests_pos[, 2]))
} 
setwd("/home/simon/PycharmProjects/robert_sql/")
write.table(slide_mat_neg, file = "slide_data_neg.txt", append = FALSE, quote = TRUE, sep = " ",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")
write.table(slide_mat_pos, file = "slide_data_pos.txt", append = FALSE, quote = TRUE, sep = " ",
             eol = "\n", na = "NA", dec = ".", row.names = TRUE,
             col.names = TRUE, qmethod = c("escape", "double"),
             fileEncoding = "")


df_neg = data.frame(slide_mat_neg)
df_pos = data.frame(slide_mat_pos)

slide_mat_neg1 = rapply(df_neg, f=function(x) ifelse(is.nan(x),0,x), how="replace" )
slide_mat_pos1 = rapply(df_pos, f=function(x) ifelse(is.nan(x),0,x), how="replace" )

write.table(slide_mat_neg1, file = "slide_data_neg1.txt", append = FALSE, quote = TRUE, sep = " ",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")
write.table(slide_mat_pos1, file = "slide_data_pos1.txt", append = FALSE, quote = TRUE, sep = " ",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

# the follwing is an R l-bfgs Test sadly there is no diretly implemented fitting methond use later for getting cofindece and
# cross validation code is working basically creates filter matrices for the object and splits them in training and test set
# shuffling was ommited since this was only a test.

# split_matrix = rbind(matrix(0, floor(length(class_vec_pos)*0.8), 12), matrix(1, floor(length(class_vec_pos)*0.2+1), 12))
# X_test_pos = slide_mat_pos1 * split_matrix
# X_test_pos =  X_test_pos[rowSums(X_test_pos != 0) > 0, ]
# 
# split_matrix = rbind(matrix(0, floor(length(class_vec_neg)*0.8), 12), matrix(1, floor(length(class_vec_neg)*0.2+1), 12))
# X_test_neg = slide_mat_neg1 * split_matrix
# X_test_neg = X_test_neg[rowSums(X_test_neg != 0) > 0, ]
# 
# split_matrix = rbind(matrix(1, floor(length(class_vec_pos)*0.8), 12), matrix(0, floor(length(class_vec_pos)*0.2+1), 12))
# X_train_pos = slide_mat_pos1 * split_matrix
# X_train_pos = X_train_pos[rowSums(X_train_pos !=0 ) > 0, ]
# 
# split_matrix = rbind(matrix(1, floor(length(class_vec_neg)*0.8), 12), matrix(0, floor(length(class_vec_neg)*0.2+1), 12))
# X_train_neg = slide_mat_neg1 * split_matrix
# X_train_neg = X_train_neg[rowSums(X_train_neg !=0 ) > 0,]
# 
# split_vec_test_pos = c(rep(0, floor(length(class_vec_pos)*0.8)), rep(1, floor(length(class_vec_pos)*0.2) + 1))
# split_vec_train_pos = c(rep(1, floor(length(class_vec_pos)*0.8)), rep(0, floor(length(class_vec_pos)*0.2) + 1))
# split_vec_train_neg = c(rep(1, floor(length(class_vec_neg)*0.8)), rep(0, floor(length(class_vec_neg)*0.2) + 1))
# 
# # 0: Her2 - ; 1: Her2 +
# 
# y_train_pos = class_vec_pos * split_vec_train_pos
# y_train_pos = y_train_pos[y_train_pos != 0]
# y_train_neg = class_vec_neg + split_vec_train_neg
# y_train_neg1 = y_train_neg[y_train_neg != 0]
# y_test_neg = y_train_neg[y_train_neg == 0]
# y_train_neg = y_train_neg1 - 1
# 
# y_test_pos = class_vec_pos * split_vec_test_pos
# y_test_pos = y_test_pos[y_test_pos != 0]
# 
# # X_train = (rbind(X_train_pos, X_train_neg))
# X_test = t(rbind(X_test_pos, X_test_neg))
# y = c(y_train_pos, y_train_neg)
# #   results = fastLR(
# #     X_train,
# #     y,
# #     start = rep(0, nrow(X_train)),
# #     eps_f = 1e-08,
# #     eps_g = 1e-05,
# #     maxit = 300
# #   )
# # results
