

if (!require(psych)) {
  install.packages("psych")
}
if (!require(FSA)) {
  install.packages("FSA")
}
if (!require(rcompanion)) {
  install.packages("rcompanion")
}
if (!require(coin)) {
  install.packages("coin")
}
library(comprehenr)
library(psych)



lc_swat_p<-read.csv("lc_swat_p.csv")
lc_swat_r<-read.csv("lc_swat_r.csv")
lc_swat_f1<-read.csv("lc_swat_f1.csv")
mg_swat_p<-read.csv("mg_swat_p.csv")
mg_swat_r<-read.csv("mg_swat_r.csv")
mg_swat_f1<-read.csv("mg_swat_f1.csv")
attain_swat_p<-read.csv("attain_swat_p.csv")
attain_swat_r<-read.csv("attain_swat_r.csv")
attain_swat_f1<-read.csv("attain_swat_f1.csv")
lattice_swat_p<-read.csv("lattice_swat_p.csv")
lattice_swat_r<-read.csv("lattice_swat_r.csv")
lattice_swat_f1<-read.csv("lattice_swat_f1.csv")
lattice_s1_swat_p<-read.csv("lattice_s1_swat_p.csv")
lattice_s1_swat_r<-read.csv("lattice_s1_swat_r.csv")
lattice_s1_swat_f1<-read.csv("lattice_s1_swat_f1.csv")
lc_wadi_p<-read.csv("lc_wadi_p.csv")
lc_wadi_r<-read.csv("lc_wadi_r.csv")
lc_wadi_f1<-read.csv("lc_wadi_f1.csv")
mg_wadi_p<-read.csv("mg_wadi_p.csv")
mg_wadi_r<-read.csv("mg_wadi_r.csv")
mg_wadi_f1<-read.csv("mg_wadi_f1.csv")
attain_wadi_p<-read.csv("attain_wadi_p.csv")
attain_wadi_r<-read.csv("attain_wadi_r.csv")
attain_wadi_f1<-read.csv("attain_wadi_f1.csv")
lattice_wadi_p<-read.csv("lattice_wadi_p.csv")
lattice_wadi_r<-read.csv("lattice_wadi_r.csv")
lattice_wadi_f1<-read.csv("lattice_wadi_f1.csv")
lattice_s1_wadi_p<-read.csv("lattice_s1_wadi_p.csv")
lattice_s1_wadi_r<-read.csv("lattice_s1_wadi_r.csv")
lattice_s1_wadi_f1<-read.csv("lattice_s1_wadi_f1.csv")
lc_batadal_p<-read.csv("lc_batadal_p.csv")
lc_batadal_r<-read.csv("lc_batadal_r.csv")
lc_batadal_f1<-read.csv("lc_batadal_f1.csv")
mg_batadal_p<-read.csv("mg_batadal_p.csv")
mg_batadal_r<-read.csv("mg_batadal_r.csv")
mg_batadal_f1<-read.csv("mg_batadal_f1.csv")
attain_batadal_p<-read.csv("attain_batadal_p.csv")
attain_batadal_r<-read.csv("attain_batadal_r.csv")
attain_batadal_f1<-read.csv("attain_batadal_f1.csv")
lattice_batadal_p<-read.csv("lattice_batadal_p.csv")
lattice_batadal_r<-read.csv("lattice_batadal_r.csv")
lattice_batadal_f1<-read.csv("lattice_batadal_f1.csv")
lattice_s1_batadal_p<-read.csv("lattice_s1_batadal_p.csv")
lattice_s1_batadal_r<-read.csv("lattice_s1_batadal_r.csv")
lattice_s1_batadal_f1<-read.csv("lattice_s1_batadal_f1.csv")
lc_phm_p<-read.csv("lc_phm_p.csv")
lc_phm_r<-read.csv("lc_phm_r.csv")
lc_phm_f1<-read.csv("lc_phm_f1.csv")
mg_phm_p<-read.csv("mg_phm_p.csv")
mg_phm_r<-read.csv("mg_phm_r.csv")
mg_phm_f1<-read.csv("mg_phm_f1.csv")
attain_phm_p<-read.csv("attain_phm_p.csv")
attain_phm_r<-read.csv("attain_phm_r.csv")
attain_phm_f1<-read.csv("attain_phm_f1.csv")
lattice_phm_p<-read.csv("lattice_phm_p.csv")
lattice_phm_r<-read.csv("lattice_phm_r.csv")
lattice_phm_f1<-read.csv("lattice_phm_f1.csv")
lattice_s1_phm_p<-read.csv("lattice_s1_phm_p.csv")
lattice_s1_phm_r<-read.csv("lattice_s1_phm_r.csv")
lattice_s1_phm_f1<-read.csv("lattice_s1_phm_f1.csv")
lc_gas_p<-read.csv("lc_gas_p.csv")
lc_gas_r<-read.csv("lc_gas_r.csv")
lc_gas_f1<-read.csv("lc_gas_f1.csv")
mg_gas_p<-read.csv("mg_gas_p.csv")
mg_gas_r<-read.csv("mg_gas_r.csv")
mg_gas_f1<-read.csv("mg_gas_f1.csv")
attain_gas_p<-read.csv("attain_gas_p.csv")
attain_gas_r<-read.csv("attain_gas_r.csv")
attain_gas_f1<-read.csv("attain_gas_f1.csv")
lattice_gas_p<-read.csv("lattice_gas_p.csv")
lattice_gas_r<-read.csv("lattice_gas_r.csv")
lattice_gas_f1<-read.csv("lattice_gas_f1.csv")
lattice_s1_gas_p<-read.csv("lattice_s1_gas_p.csv")
lattice_s1_gas_r<-read.csv("lattice_s1_gas_r.csv")
lattice_s1_gas_f1<-read.csv("lattice_s1_gas_f1.csv")



# =================================lattice lc=============================================
data<-lattice_swat_p-lc_swat_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
              )
print("lattice lc swat p")
print(test_result)


data<-lattice_swat_r-lc_swat_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc swat r")
print(test_result)

data<-lattice_swat_f1-lc_swat_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc swat f1")
print(test_result)

data<-lattice_wadi_p-lc_wadi_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc wadi p")
print(test_result)

data<-lattice_wadi_r-lc_wadi_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc wadi r")
print(test_result)

data<-lattice_wadi_f1-lc_wadi_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc wadi f1")
print(test_result)

data<-lattice_batadal_p-lc_batadal_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc batadal p")
print(test_result)

data<-lattice_batadal_r-lc_batadal_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc batadal r")
print(test_result)

data<-lattice_batadal_f1-lc_batadal_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc batadal f1")
print(test_result)

data<-lattice_phm_p-lc_phm_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc phm p")
print(test_result)

data<-lattice_phm_r-lc_phm_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc phm r")
print(test_result)

data<-lattice_phm_f1-lc_phm_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc phm f1")
print(test_result)

data<-lattice_gas_p-lc_gas_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc gas p")
print(test_result)

data<-lattice_gas_r-lc_gas_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc gas r")
print(test_result)


data<-lattice_gas_f1-lc_gas_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc gas f1")
print(test_result)


# =================================lattice mg=============================================
data<-lattice_swat_p-mg_swat_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg swat p")
print(test_result)


data<-lattice_swat_r-mg_swat_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg swat r")
print(test_result)

data<-lattice_swat_f1-mg_swat_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg swat f1")
print(test_result)

data<-lattice_wadi_p-mg_wadi_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg wadi p")
print(test_result)

data<-lattice_wadi_r-mg_wadi_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg wadi r")
print(test_result)

data<-lattice_wadi_f1-mg_wadi_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg wadi f1")
print(test_result)

data<-lattice_batadal_p-mg_batadal_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg batadal p")
print(test_result)

data<-lattice_batadal_r-mg_batadal_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg batadal r")
print(test_result)

data<-lattice_batadal_f1-mg_batadal_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg batadal f1")
print(test_result)

data<-lattice_phm_p-mg_phm_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg phm p")
print(test_result)

data<-lattice_phm_r-mg_phm_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg phm r")
print(test_result)

data<-lattice_phm_f1-mg_phm_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg phm f1")
print(test_result)

data<-lattice_gas_p-mg_gas_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg gas p")
print(test_result)

data<-lattice_gas_r-mg_gas_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg gas r")
print(test_result)


data<-lattice_gas_f1-mg_gas_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg gas f1")
print(test_result)


# =================================lattice attain=============================================
data<-lattice_swat_p-attain_swat_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain swat p")
print(test_result)


data<-lattice_swat_r-attain_swat_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain swat r")
print(test_result)

data<-lattice_swat_f1-attain_swat_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain swat f1")
print(test_result)

data<-lattice_wadi_p-attain_wadi_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain wadi p")
print(test_result)

data<-lattice_wadi_r-attain_wadi_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain wadi r")
print(test_result)

data<-lattice_wadi_f1-attain_wadi_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain wadi f1")
print(test_result)

data<-lattice_batadal_p-attain_batadal_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain batadal p")
print(test_result)

data<-lattice_batadal_r-attain_batadal_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain batadal r")
print(test_result)

data<-lattice_batadal_f1-attain_batadal_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain batadal f1")
print(test_result)

data<-lattice_phm_p-attain_phm_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain phm p")
print(test_result)

data<-lattice_phm_r-attain_phm_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain phm r")
print(test_result)

data<-lattice_phm_f1-attain_phm_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain phm f1")
print(test_result)

data<-lattice_gas_p-attain_gas_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain gas p")
print(test_result)

data<-lattice_gas_r-attain_gas_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain gas r")
print(test_result)


data<-lattice_gas_f1-attain_gas_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain gas f1")
print(test_result)


# =================================lattice s1=============================================
data<-lattice_swat_p-lattice_s1_swat_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 swat p")
print(test_result)


data<-lattice_swat_r-lattice_s1_swat_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 swat r")
print(test_result)

data<-lattice_swat_f1-lattice_s1_swat_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 swat f1")
print(test_result)

data<-lattice_wadi_p-lattice_s1_wadi_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 wadi p")
print(test_result)

data<-lattice_wadi_r-lattice_s1_wadi_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 wadi r")
print(test_result)

data<-lattice_wadi_f1-lattice_s1_wadi_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 wadi f1")
print(test_result)

data<-lattice_batadal_p-lattice_s1_batadal_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 batadal p")
print(test_result)

data<-lattice_batadal_r-lattice_s1_batadal_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 batadal r")
print(test_result)

data<-lattice_batadal_f1-lattice_s1_batadal_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 batadal f1")
print(test_result)

data<-lattice_phm_p-lattice_s1_phm_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 phm p")
print(test_result)

data<-lattice_phm_r-lattice_s1_phm_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 phm r")
print(test_result)

data<-lattice_phm_f1-lattice_s1_phm_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 phm f1")
print(test_result)

data<-lattice_gas_p-lattice_s1_gas_p
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 gas p")
print(test_result)

data<-lattice_gas_r-lattice_s1_gas_r
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 gas r")
print(test_result)


data<-lattice_gas_f1-lattice_s1_gas_f1
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lattice_s1 gas f1")
print(test_result)


# ================================execution time testing ======================


et_lc_swat<-read.csv("et_lc_swat.csv")
et_lc_wadi<-read.csv("et_lc_wadi.csv")
et_lc_batadal<-read.csv("et_lc_batadal.csv")
et_lc_phm<-read.csv("et_lc_phm.csv")
et_lc_gas<-read.csv("et_lc_gas.csv")
et_mg_swat<-read.csv("et_mg_swat.csv")
et_mg_wadi<-read.csv("et_mg_wadi.csv")
et_mg_batadal<-read.csv("et_mg_batadal.csv")
et_mg_phm<-read.csv("et_mg_phm.csv")
et_mg_gas<-read.csv("et_mg_gas.csv")
et_attain_swat<-read.csv("et_attain_swat.csv")
et_attain_wadi<-read.csv("et_attain_wadi.csv")
et_attain_batadal<-read.csv("et_attain_batadal.csv")
et_attain_phm<-read.csv("et_attain_phm.csv")
et_attain_gas<-read.csv("et_attain_gas.csv")
et_lattice_swat<-read.csv("et_lattice_swat.csv")
et_lattice_wadi<-read.csv("et_lattice_wadi.csv")
et_lattice_batadal<-read.csv("et_lattice_batadal.csv")
et_lattice_phm<-read.csv("et_lattice_phm.csv")
et_lattice_gas<-read.csv("et_lattice_gas.csv")


et_es_mat<-matrix(0.5,nrow=5,ncol=3)
es<-VD.A(et_lattice_swat,et_lc_swat,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[1,1]=es$estimate

es<-VD.A(et_lattice_swat,et_mg_swat,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[1,2]=es$estimate

es<-VD.A(et_lattice_swat,et_attain_swat,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[1,3]=es$estimate

es<-VD.A(et_lattice_wadi,et_lc_wadi,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[2,1]=es$estimate

es<-VD.A(et_lattice_wadi,et_mg_wadi,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[2,2]=es$estimate

es<-VD.A(et_lattice_wadi,et_attain_wadi,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[2,3]=es$estimate

es<-VD.A(et_lattice_batadal,et_lc_batadal,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[3,1]=es$estimate

es<-VD.A(et_lattice_batadal,et_mg_batadal,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[3,2]=es$estimate

es<-VD.A(et_lattice_batadal,et_attain_batadal,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[3,3]=es$estimate

es<-VD.A(et_lattice_phm,et_lc_phm,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[4,1]=es$estimate

es<-VD.A(et_lattice_phm,et_mg_phm,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[4,2]=es$estimate

es<-VD.A(et_lattice_phm,et_attain_phm,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[4,3]=es$estimate

es<-VD.A(et_lattice_gas,et_lc_gas,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[5,1]=es$estimate

es<-VD.A(et_lattice_gas,et_mg_gas,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[5,2]=es$estimate

es<-VD.A(et_lattice_gas,et_attain_gas,paired=TRUE)
print(class(es))
print(es$estimate)
et_es_mat[5,3]=es$estimate
print(et_es_mat)

data<-et_lattice_swat- et_lc_swat 
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc swat")
print(test_result)

data<-et_lattice_wadi- et_lc_wadi
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc wadi")
print(test_result)

data<-et_lattice_batadal- et_lc_batadal
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc batadal")
print(test_result)

data<-et_lattice_phm- et_lc_phm
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc phm")
print(test_result)

data<-et_lattice_gas- et_lc_gas
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice lc gas")
print(test_result)


data<-et_lattice_swat- et_mg_swat 
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg swat")
print(test_result)

data<-et_lattice_wadi- et_mg_wadi
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg wadi")
print(test_result)

data<-et_lattice_batadal- et_mg_batadal
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg batadal")
print(test_result)

data<-et_lattice_phm- et_mg_phm
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg phm")
print(test_result)

data<-et_lattice_gas- et_mg_gas
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice mg gas")
print(test_result)

data<-et_lattice_swat- et_attain_swat 
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain swat")
print(test_result)

data<-et_lattice_wadi- et_attain_wadi
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain wadi")
print(test_result)

data<-et_lattice_batadal- et_attain_batadal
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain batadal")
print(test_result)

data<-et_lattice_phm- et_attain_phm
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain phm")
print(test_result)

data<-et_lattice_gas- et_attain_gas
test_result <-
  wilcox.test(data,
              mu = 0,
              conf.int = TRUE,
              conf.level = 0.99
  )
print("lattice attain gas")
print(test_result)
#save.image(file="testing.rdata")
load(file="testing.rdata")

######################################### effect size ####################
if (!require(effsize)) {
  install.packages("effsize")
}
library(effsize)
lc_es_mat=matrix(0.5,nrow=3,ncol=5)
mg_es_mat=matrix(0.5,nrow=3,ncol=5)
attain_es_mat=matrix(0.5,nrow=3,ncol=5)
lattice_s1_es_mat=matrix(0.5,nrow=3,ncol=5)
# lc
print("lc")
es<-VD.A(lattice_swat_p,lc_swat_p,paired=TRUE)
print(class(es))
print(es$estimate)
lc_es_mat[1,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_r,lc_swat_r,paired=TRUE)
lc_es_mat[2,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_f1,lc_swat_f1,paired=TRUE)
lc_es_mat[3,1]=es$estimate
print(es)
es<-VD.A(lattice_wadi_p,lc_wadi_p,paired=TRUE)
lc_es_mat[1,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_r,lc_wadi_r,paired=TRUE)
lc_es_mat[2,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_f1,lc_wadi_f1,paired=TRUE)
lc_es_mat[3,2]=es$estimate
print(es)
es<-VD.A(lattice_batadal_p,lc_batadal_p,paired=TRUE)
lc_es_mat[1,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_r,lc_batadal_r,paired=TRUE)
lc_es_mat[2,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_f1,lc_batadal_f1,paired=TRUE)
lc_es_mat[3,3]=es$estimate
print(es)
es<-VD.A(lattice_phm_p,lc_phm_p,paired=TRUE)
lc_es_mat[1,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_r,lc_phm_r,paired=TRUE)
lc_es_mat[2,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_f1,lc_phm_f1,paired=TRUE)
lc_es_mat[3,4]=es$estimate
print(es)
es<-VD.A(lattice_gas_p,lc_gas_p,paired=TRUE)
lc_es_mat[1,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_r,lc_gas_r,paired=TRUE)
lc_es_mat[2,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_f1,lc_gas_f1,paired=TRUE)
lc_es_mat[3,5]=es$estimate
print(es)
print(lc_es_mat)
#mg

es<-VD.A(lattice_swat_p,mg_swat_p,paired=TRUE)
mg_es_mat[1,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_r,mg_swat_r,paired=TRUE)
mg_es_mat[2,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_f1,mg_swat_f1,paired=TRUE)
mg_es_mat[3,1]=es$estimate
print(es)
es<-VD.A(lattice_wadi_p,mg_wadi_p,paired=TRUE)
mg_es_mat[1,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_r,mg_wadi_r,paired=TRUE)
mg_es_mat[2,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_f1,mg_wadi_f1,paired=TRUE)
mg_es_mat[3,2]=es$estimate
print(es)
es<-VD.A(lattice_batadal_p,mg_batadal_p,paired=TRUE)
mg_es_mat[1,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_r,mg_batadal_r,paired=TRUE)
mg_es_mat[2,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_f1,mg_batadal_f1,paired=TRUE)
mg_es_mat[3,3]=es$estimate
print(es)
es<-VD.A(lattice_phm_p,mg_phm_p,paired=TRUE)
mg_es_mat[1,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_r,mg_phm_r,paired=TRUE)
mg_es_mat[2,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_f1,mg_phm_f1,paired=TRUE)
mg_es_mat[3,4]=es$estimate
print(es)
es<-VD.A(lattice_gas_p,mg_gas_p,paired=TRUE)
mg_es_mat[1,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_r,mg_gas_r,paired=TRUE)
mg_es_mat[2,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_f1,mg_gas_f1,paired=TRUE)
mg_es_mat[3,5]=es$estimate
print(es)
#ATTAIN
es<-VD.A(lattice_swat_p,attain_swat_p,paired=TRUE)
attain_es_mat[1,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_r,attain_swat_r,paired=TRUE)
attain_es_mat[2,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_f1,attain_swat_f1,paired=TRUE)
attain_es_mat[3,1]=es$estimate
print(es)
es<-VD.A(lattice_wadi_p,attain_wadi_p,paired=TRUE)
attain_es_mat[1,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_r,attain_wadi_r,paired=TRUE)
attain_es_mat[2,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_f1,attain_wadi_f1,paired=TRUE)
attain_es_mat[3,2]=es$estimate
print(es)
es<-VD.A(lattice_batadal_p,attain_batadal_p,paired=TRUE)
attain_es_mat[1,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_r,attain_batadal_r,paired=TRUE)
attain_es_mat[2,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_f1,attain_batadal_f1,paired=TRUE)
attain_es_mat[3,3]=es$estimate
print(es)
es<-VD.A(lattice_phm_p,attain_phm_p,paired=TRUE)
attain_es_mat[1,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_r,attain_phm_r,paired=TRUE)
attain_es_mat[2,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_f1,attain_phm_f1,paired=TRUE)
attain_es_mat[3,4]=es$estimate
print(es)
es<-VD.A(lattice_gas_p,attain_gas_p,paired=TRUE)
attain_es_mat[1,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_r,attain_gas_r,paired=TRUE)
attain_es_mat[2,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_f1,attain_gas_f1,paired=TRUE)
attain_es_mat[3,5]=es$estimate
print(es)
print(attain_es_mat)
heatmap(attain_es_mat)

# s1

es<-VD.A(lattice_swat_p,lattice_s1_swat_p,paired=TRUE)
lattice_s1_es_mat[1,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_r,lattice_s1_swat_r,paired=TRUE)
lattice_s1_es_mat[2,1]=es$estimate
print(es)
es<-VD.A(lattice_swat_f1,lattice_s1_swat_f1,paired=TRUE)
lattice_s1_es_mat[3,1]=es$estimate
print(es)
es<-VD.A(lattice_wadi_p,lattice_s1_wadi_p,paired=TRUE)
lattice_s1_es_mat[1,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_r,lattice_s1_wadi_r,paired=TRUE)
lattice_s1_es_mat[2,2]=es$estimate
print(es)
es<-VD.A(lattice_wadi_f1,lattice_s1_wadi_f1,paired=TRUE)
lattice_s1_es_mat[3,2]=es$estimate
print(es)
es<-VD.A(lattice_batadal_p,lattice_s1_batadal_p,paired=TRUE)
lattice_s1_es_mat[1,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_r,lattice_s1_batadal_r,paired=TRUE)
lattice_s1_es_mat[2,3]=es$estimate
print(es)
es<-VD.A(lattice_batadal_f1,lattice_s1_batadal_f1,paired=TRUE)
lattice_s1_es_mat[3,3]=es$estimate
print(es)
es<-VD.A(lattice_phm_p,lattice_s1_phm_p,paired=TRUE)
lattice_s1_es_mat[1,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_r,lattice_s1_phm_r,paired=TRUE)
lattice_s1_es_mat[2,4]=es$estimate
print(es)
es<-VD.A(lattice_phm_f1,lattice_s1_phm_f1,paired=TRUE)
lattice_s1_es_mat[3,4]=es$estimate
print(es)
es<-VD.A(lattice_gas_p,lattice_s1_gas_p,paired=TRUE)
lattice_s1_es_mat[1,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_r,lattice_s1_gas_r,paired=TRUE)
lattice_s1_es_mat[2,5]=es$estimate
print(es)
es<-VD.A(lattice_gas_f1,lattice_s1_gas_f1,paired=TRUE)
lattice_s1_es_mat[3,5]=es$estimate
print(es)
print(lattice_s1_es_mat)
heatmap(lattice_s1_es_mat)

# boxplot
par(mfrow=c(5,3))
boxplot(lc_swat_p,mg_swat_p,attain_swat_p,lattice_swat_p,main="Precision on SWaT", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_swat_r,mg_swat_r,attain_swat_r,lattice_swat_r,main="Recall on SWaT", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_swat_f1,mg_swat_f1,attain_swat_f1,lattice_swat_f1,main="F1 on SWaT", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_wadi_p,mg_wadi_p,attain_wadi_p,lattice_wadi_p,main="Precision on WADI", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_wadi_r,mg_wadi_r,attain_wadi_r,lattice_wadi_r,main="Recall on WADI", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_wadi_f1,mg_wadi_f1,attain_wadi_f1,lattice_wadi_f1,main="F1 on WADI", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_batadal_p,mg_batadal_p,attain_batadal_p,lattice_batadal_p,main="Precision on BATADAL", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_batadal_r,mg_batadal_r,attain_batadal_r,lattice_batadal_r,main="Recall on BATADAL", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_batadal_f1,mg_batadal_f1,attain_batadal_f1,lattice_batadal_f1,main="Precision on BATADAL", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
#par(mfrow=c(3,3))
boxplot(lc_phm_p,mg_phm_p,attain_phm_p,lattice_phm_p,main="Precision on PHM", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_phm_r,mg_phm_r,attain_phm_r,lattice_phm_r,main="Recall on PHM", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_phm_f1,mg_phm_f1,attain_phm_f1,lattice_phm_f1,main="F1 on PHM", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_gas_p,mg_gas_p,attain_gas_p,lattice_gas_p,main="Precision on GAS", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_gas_r,mg_gas_r,attain_gas_r,lattice_gas_r,main="Recall on GAS", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))
boxplot(lc_gas_f1,mg_gas_f1,attain_gas_f1,lattice_gas_f1,main="F1 on GAS", names = c("lSTM-CUSUM","MAD-GAN","ATTAIN","LATTICE"))

# heatmap
library(RColorBrewer)
coul <- colorRampPalette(brewer.pal(8, "Greys"))(25)
heatmap(lc_es_mat,col=coul,Rowv=NA,Colv=NA)
heatmap(mg_es_mat,col=coul,Rowv=NA,Colv=NA)
heatmap(attain_es_mat,col=coul,Rowv=NA,Colv=NA)

if (!require(ggplot2)) {
  install.packages("ggplot2")
}
if (!require(reshape2)) {
  install.packages("reshape2")
}
if (!require(hrbrthemes)) {
  install.packages("hrbrthemes")
}
library(ggplot2)
library(reshape2)
library(hrbrthemes)

# Dummy data


# Give extreme colors:
if (!require(gridExtra)) {
  install.packages("gridExtra")
}
require(gridExtra)

data<-data.frame(lc_es_mat)
names(data)<-c("SWaT","WADI","BATADAL","PHM","GAS")
data["metric"]=c("precision","recall","f1")
print(data)
data<-reshape2::melt(data,id.vars=c("metric"))
print(data)
p1<-ggplot(data,aes(x=variable,y=metric,fill=value))  + 
  geom_tile() + 
  scale_fill_gradient2(low="white",high="black",limits=c(0,1))+
  theme_ipsum()+
  theme(axis.title.x=element_blank())+
  ggtitle("Comparsion with LSTM-CUSUM")


data<-data.frame(mg_es_mat)
names(data)<-c("SWaT","WADI","BATADAL","PHM","GAS")
data["metric"]=c("precision","recall","f1")
print(data)
data<-reshape2::melt(data,id.vars=c("metric"))
print(data)
p2<-ggplot(data,aes(x=variable,y=metric,fill=value))  + 
  geom_tile() + 
  scale_fill_gradient2(low="white",high="black",limits=c(0,1))+
  theme_ipsum()+
  theme(axis.title.x=element_blank())+
  ggtitle("Comparsion with MAD-GAN")


data<-data.frame(attain_es_mat)
names(data)<-c("SWaT","WADI","BATADAL","PHM","GAS")
data["metric"]=c("precision","recall","f1")
print(data)
data<-reshape2::melt(data,id.vars=c("metric"))
print(data)
p3<-ggplot(data,aes(x=variable,y=metric,fill=value))  + 
  geom_tile() + 
  scale_fill_gradient2(low="white",high="black",limits=c(0,1))+
  theme_ipsum()+
  theme(axis.title.x=element_blank())+
  ggtitle("Comparsion with ATTAIN")

p1
p2
p3
grid.arrange(p1,p2,p3,ncol=1)
