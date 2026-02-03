

library(mcclust.ext)
library(RcppCNPy)
#x <- npyLoad("Muestra.npy")
x <- read.csv("EarthquakesR.csv",header=FALSE)

setwd('/Users/isaias/Desktop/Poisson_Earthquakes/')
ex2.draw2=read.csv("LabelClustMexico.csv",header=FALSE,sep=",")
ex2.draw2=as.matrix(ex2.draw2)+1
psm=comp.psm(ex2.draw2)
ex2.B=minbinder.ext(psm,ex2.draw2,method=("all"),include.greedy=TRUE)


summary(ex2.B)
plot(ex2.B,data=data.frame(x[,1:2]))

ex2.VI=minVI(psm,ex2.draw2,method=("all"),include.greedy=TRUE)
plot(ex2.VI,data=data.frame(x[,1:2]))

x[,"Longitude"]
x[,"Latitude"]
















