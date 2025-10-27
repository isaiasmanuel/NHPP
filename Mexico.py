#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:46:14 2025

@author: isaias
"""

#Funciones, Simulado, Mexico

import os 

os.chdir('/Users/isaias/Desktop/Poisson_Earthquakes/')

from Funciones import *

#Simulados, Funciones, Mexico

####################





Earthquakes=pd.read_csv('./Earthquakes.csv')
Trench=pd.read_csv("./trench_cocos.csv", delimiter=',')

Earthquakes.Year.min()

Earthquakes.loc[0]["Strike 2"]


xlim,Xlim=-105.5,-96.5
ylim,Ylim=15,19.5


# xlim,Xlim=-119.5,-86.5
# ylim,Ylim=11,34.5


Earthquakes=Earthquakes[~Earthquakes["Strike 2"].isnull()]
Earthquakes=Earthquakes.loc[ (Earthquakes["Longitude"]<Xlim) & (Earthquakes["Longitude"]>xlim) &  (Earthquakes["Latitude"]<Ylim)&  (Earthquakes["Latitude"]>ylim) ]
#Earthquakes=Earthquakes.loc[Earthquakes["Depth"]!="Sh"]
#Earthquakes["Depth"]=Earthquakes["Depth"].astype(float) 
Earthquakes=Earthquakes.loc[Earthquakes["Depth"]<=100]
Earthquakes=Earthquakes.loc[Earthquakes["Mw*"]>=4.]
# Earthquakes=Earthquakes.loc[Earthquakes["Year"]>=1980]
Earthquakes=Earthquakes.loc[Earthquakes["Year"]>=2000]
# Earthquakes=Earthquakes.loc[(Earthquakes["Rake 1"]>45) & (Earthquakes["Rake 1"]<134)]
Earthquakes=Earthquakes.reset_index(drop=True)


# mindate=datetime(2000, 1,1)
mindate=datetime(2000, 1,1)
Maxdate=datetime(2015, 12,31)


len(Earthquakes)

sectoday=86400

for i in range(len(Earthquakes)):
  if i==0:
    Fechas=datetime(Earthquakes["Year"][i], Earthquakes["Month"][i], Earthquakes["Day"][i], int(Earthquakes["Hour"][i]), int(Earthquakes["Minute"][i]), int(Earthquakes["Second"][i]), int(str(Earthquakes["Second"][i])[-1])*100000)
  else :
    Fechas=np.hstack((Fechas,datetime(Earthquakes["Year"][i], Earthquakes["Month"][i], Earthquakes["Day"][i], int(Earthquakes["Hour"][i]), int(Earthquakes["Minute"][i]), int(Earthquakes["Second"][i]), int(str(Earthquakes["Second"][i])[-1])*100000)))


Diferencias=np.zeros(len(Fechas))
for i in range(len(Diferencias)):
    Diferencias[i]=(Fechas[i]-mindate).total_seconds()


Tmax=(Maxdate-mindate).total_seconds()/sectoday
Tmin=0
Diferencias=Diferencias/sectoday





# Tmax=np.max(Earthquakes["Year"])
# Tmin=np.min(Earthquakes["Year"])


############################

SimPoisson=(Earthquakes[["Longitude","Latitude"]]).to_numpy()
SimPoisson=np.hstack((SimPoisson,np.reshape(Diferencias,(len(Diferencias),1))))

#######Hyperparameters
#Particiones T
pT=4
CutPoint=np.arange(Tmin,Tmax+(Tmax-Tmin)/pT,(Tmax-Tmin)/pT)
#Aproximacion finita
ClustTiempo=3 #2 cuando son pocos y 4 cuando no
L=pT*ClustTiempo
# print(len(Earthquakes)/L)

cmap = plt.cm.get_cmap('viridis', L)

#NIW
nu=6 #Chackrabarti usa nu=2, lam=0.01,Sprior Identity Mu=0,0
# 2/(nu-2-1)
# np.sqrt(2*2**2/((nu-2-1)**2)*((nu-2-3)**2))
# 2*2**2/((nu-2-1)**2)*((nu-2-3)**2)
Sprior=np.diag((2,2)) #Parece funcionar nu=10, Sprior=10I, mu=1,1, lam=2
mu=np.array((-102,17))
lam=0.01#0.01
#alpha0
alpha0=1
#
pgamma=1#len(Earthquakes)/Tmax #70
k=0.01
GammaPrior=gamma(a=pgamma*k,scale=1/k)
GammaPrior.rvs()
IW=invwishart(df=nu,scale=Sprior)
Ssim=IW.rvs()
Gaus=multivariate_normal(mean=mu, cov=Ssim/lam)
muSim=Gaus.rvs()
# print(muSim,Ssim)

SimPoisson=np.hstack((SimPoisson,np.zeros((len(SimPoisson),1))))
for i in range(pT):    
    SimPoisson[(CutPoint[i]<SimPoisson[:,2])&(SimPoisson[:,2]<CutPoint[i+1]),3]=i

SimPoisson[:,3]
############################
Alpha=alpha0*np.ones(pT)
Beta=np.ones((pT,L))/L
Z=np.random.choice(np.arange(L),size=len(SimPoisson))
gammaParam=GammaPrior.rvs(pT)

for i in range(L):
    IW=invwishart(df=nu,scale=Sprior)
    Ssim=IW.rvs()
    Gaus=multivariate_normal(mean=mu, cov=Ssim/lam)
    muSim=np.hstack((np.random.uniform(xlim,Xlim,size=1),np.random.uniform(ylim,Ylim,size=1))) #Gaus.rvs()
    
    
    if i==0:
        Phi=[[muSim,Ssim]]
    else :
        Phi.append([muSim,Ssim])

####################



CutTime=np.arange(Tmin,Tmax+(Tmax-Tmin)/pT,(Tmax-Tmin)/pT)




for i in range(len(CutTime)-1):
    EarthquakesAux=Earthquakes.loc[ (Diferencias>=CutTime[i]) & (Diferencias<=CutTime[i+1])]
    # print(len(EarthquakesAux))
    # xlim,Xlim=-106,-96
    # ylim,Ylim=14.5,20
    plt.xlim(xlim,Xlim)    
    plt.ylim(ylim,Ylim)
    npartX=20
    npartY=20
    DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
    DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
    shape=shp.Reader("./Mexico.shp") #http://geoportal.conabio.gob.mx/metadatos/doc/html/dest_2010gw.html
    for shape in shape.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y,color="black")    

    plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"])
    plt.plot(Trench["x"],Trench["y"],color="hotpink")
    plt.xlim(DomX[0],DomX[-1])
    plt.ylim(DomY[0],DomY[-1])
    # plt.title("Interval:"+str(round(CutTime[i],2))+"-"+str(round(CutTime[i+1],2)))
    plt.title("Interval:"+str((mindate+timedelta(days=CutTime[i])))+"-"+str((mindate+timedelta(days=CutTime[i+1]))))
    plt.show()
###################



Cadena=[[Alpha,Beta,gammaParam,Z,Phi]]

# print(len(SimPoisson))
# print(np.sum(L*pT*6+pT+pT*L+L))


ChainSize=1_000_000

np.random.seed(1)




Variables((Tmax,pT,xlim,Xlim,ylim,Ylim,SimPoisson,L,alpha0,IW,Gaus,GammaPrior))


Actualizaciones=np.zeros(0)


# for contador in range(ChainSize):
#     Variables2(Cadena)
    

#     ActWeigh=np.array((1,1,1,1,1))
#     # ActWeigh=np.array((1,1,1,1,1))

#     Actualizar=np.random.choice(5, p=ActWeigh/np.sum(ActWeigh))
#     Actualizar
#     # Actualizar=contador%5
#     Actualizaciones=np.hstack((Actualizaciones,Actualizar))
#     if Actualizar==0:
#         TActualizar=np.random.choice(pT)
#         for TActualizar in range(pT):
            
#             Actual=np.copy(Cadena[-1][Actualizar])
#             Alphaprop=np.copy(Cadena[-1][Actualizar])
#             # Alphaprop[TActualizar]=Alphaprop[TActualizar]+np.random.normal(loc=0,scale=np.sqrt(0.1*Alphaprop[TActualizar]) )
#             Alphaprop[TActualizar]=Alphaprop[TActualizar]+np.random.normal(loc=0,scale=0.1)            
            
#             for i in range(100):
#                 if all(Alphaprop>0) :
#                     # rationorm=multivariate_normal.pdf(Actual[TActualizar],mean= Alphaprop[TActualizar] ,cov = 0.1*Alphaprop[TActualizar])/multivariate_normal.pdf(Alphaprop[TActualizar],mean=Cadena[-1][Actualizar][TActualizar], cov = 0.1*Actual[TActualizar])
#                     rho=min(1,a(Alphaprop,Cadena[-1][1],TActualizar)/a(Actual,Cadena[-1][1],TActualizar))#*rationorm)                
                    
#                     if np.random.choice(2, p=(1-rho,rho))==1:
#                         Actual=np.copy(Alphaprop)
    
#                 Alphaprop[TActualizar]=Actual[TActualizar]+np.random.normal(loc=0,scale=0.1)
                    
    
#             Cadena.append([Actual,Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])
                

#     if Actualizar==1:
#         probaBetaT=np.ones(pT)
#         # probaBetaT[0]=10        
#         TActualizar=np.random.choice(pT, p=probaBetaT/np.sum(probaBetaT))
        
#         for TActualizar in range(pT):
#             if TActualizar==0:
#                 Actual=Cadena[-1][Actualizar][TActualizar,:]
#                 ActualCompleto=Cadena[-1][Actualizar]
#                 for ssalt in range(100):
                    
#                     betaprop,qratio=SALT(Actual)
#                     Betaprop=np.copy(Cadena[-1][Actualizar])
#                     Betaprop[TActualizar]=betaprop
                    
#                     if all(Betaprop[TActualizar]>0):
#                         rho=min(1,qratio*b1(Cadena[-1][0],Betaprop,Cadena[-1][3])/b1(Cadena[-1][0],ActualCompleto,Cadena[-1][3]  ) )
#                     else :
#                         rho=0
                        
#                     if np.random.choice(2, p=(1-rho,rho))==1:
#                         Actual=np.copy(betaprop)
#                         ActualCompleto=np.copy(Betaprop)
                    
#                 Cadena.append([Cadena[-1][0],ActualCompleto,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])
#                 # else :
#                 #     Cadena.append(Cadena[-1])
#             if TActualizar!=0:
#                 m1l=np.zeros(L)
#                 for i in range(L):
#                     m1l[i]=np.sum(Z[SimPoisson[:,3]==TActualizar]==i)
#                     # m1l[i]=np.sum(Z==i)
#                 Betaprop=np.copy(Cadena[-1][Actualizar])
#                 Betaprop[TActualizar]=dirichlet.rvs(m1l+Cadena[-1][0][TActualizar]*Cadena[-1][1][TActualizar-1],  size=1)
               
#                 if all(Betaprop[TActualizar]>0):
#                     Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
                    
#                 elif not all(Betaprop[TActualizar]>0):
#                     # Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
#                     Betaprop[TActualizar]=Betaprop[TActualizar]+np.finfo(float).eps
#                     Betaprop[TActualizar]=Betaprop[TActualizar]/np.sum(Betaprop[TActualizar])
#                     Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
                
#                 # else :
#                 #     Cadena.append(Cadena[-1])


#     if Actualizar==2:
#         TActualizar=np.random.choice(pT)
#         gamaprop=np.copy(Cadena[-1][2])
#         # for TActualizar in range(pT):
#         gamaprop[TActualizar]=gamma.rvs( a=np.sum(SimPoisson[:,3]==TActualizar)+pgamma*k,scale=1/(k+(CutPoint[TActualizar+1]-CutPoint[TActualizar]))  ) 
#         Cadena.append([Cadena[-1][0],Cadena[-1][1],gamaprop,Cadena[-1][3],Cadena[-1][4]])
#     if Actualizar==4:
        
#         ########Idea1
#         # for TActualizar in range(pT):
#         #     ZPropuesta=np.copy(Cadena[-1][Actualizar])
#         #     for i in range(len(ZPropuesta)):
#         #         Zprob=np.zeros(L)
#         #         Zprob=mp.matrix(Zprob)
#         #         for lindex in range(L):
#         #             Zprob[lindex]=Cadena[-1][1][TActualizar][lindex]*multivariate_normal.pdf(SimPoisson[i,[0,1]], mean=Cadena[-1][4][lindex][0],cov=Cadena[-1][4][lindex][1])
#         #         dens=Zprob[:]/mp.fsum(Zprob[:])
#         #         ZPropuesta[i]=np.random.choice(L,p=dens)
#         #     # plt.scatter(SimPoisson[ZPropuesta==0][:,0],SimPoisson[ZPropuesta==0][:,1])
#         ########Idea2
#         # ZPropuesta=np.copy(Cadena[-1][Actualizar])
#         # for i in range(len(SimPoisson)):
#         #     ZPropuesta[i]=NormalZ(i)
#         ########Idea3
#         paralell= get_context("fork").Pool(cpu_count())
#         results = paralell.map(NormalZ, np.arange(len(SimPoisson)))
#         paralell.close()    
#         paralell.join()
#         ZPropuesta=np.array(results)
#         Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],ZPropuesta,Cadena[-1][4]])
        
#     if Actualizar==3:
#         LActualizar=np.random.choice(L)
#         for LActualizar in range(L):
#             MuestraAux=SimPoisson[Cadena[-1][3]==LActualizar,:2]
#             if len(MuestraAux)!=0:
#                 # plt.scatter(SimPoisson[Cadena[-1][3]==LActualizar,:2][:,0],SimPoisson[Cadena[-1][3]==LActualizar,:2][:,1])
#                 # len(MuestraAux)
#                 ybar=np.apply_along_axis(np.mean, 0,MuestraAux)
#                 # np.reshape((ybar-mu),(2,1))@np.reshape((ybar-mu),(1,2))
#                 Diferencia=(MuestraAux-ybar)
#                 S=0
#                 for i in range(len(MuestraAux)):
#                     S+=np.reshape(Diferencia[i],(2,1))@np.reshape(Diferencia[i],(1,2))
#                 naux=len(MuestraAux)
#                 mun=(mu*lam+naux*ybar)/(lam+naux)
#                 lamn=lam+naux
#                 nun=nu+naux
#                 Spost=Sprior+S+ (lam*naux)/lamn*np.reshape((ybar-mu),(2,1))@np.reshape((ybar-mu),(1,2))
#                 IW2=invwishart(df=nun,scale=Spost)
#                 muSim2=np.array((-1000,-100))
#                 while not (muSim2[0]>xlim and muSim2[0]<Xlim and muSim2[1]>ylim and muSim2[1]<Ylim):
#                     Ssim2=IW2.rvs()
#                     Gaus2=multivariate_normal(mean=mun, cov=Ssim2/lamn)
#                     muSim2=Gaus2.rvs()
                
#                 # print(muSim2,Ssim2)
#                 PhiPropuesta=copy.deepcopy(Cadena[-1][Actualizar+1])
#                 PhiPropuesta[LActualizar][0]=np.copy(muSim2)
#                 # Cadena[-1][Actualizar][LActualizar][0]
#                 PhiPropuesta[LActualizar][1]=np.copy(Ssim2)
#                 Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])            
                
                
#                 if (np.random.uniform()< min(1,mp.exp(Post([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])-Post(Cadena[-1])))):
#                     Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])
                    
#                 else :
#                     Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])     
    
#             elif len(MuestraAux)==0:
                
#                 # Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])     

#                 # ############Opcion 1 
#                 muSim2=np.array((-1000,-100))
#                 while not (muSim2[0]>xlim and muSim2[0]<Xlim and muSim2[1]>ylim and muSim2[1]<Ylim):
#                     try :
#                         Ssim2=IW.rvs()
#                         Gaus2=multivariate_normal(mean=mu, cov=Ssim2/lam)
#                         muSim2=Gaus.rvs()
#                     except:
#                         pass
#                 PhiPropuesta=copy.deepcopy(Cadena[-1][Actualizar+1])
#                 PhiPropuesta[LActualizar][0]=np.copy(muSim2)
#                 PhiPropuesta[LActualizar][1]=np.copy(Ssim2)
#                 Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])            


                
                
                
                
#                 ##############
#             # else :
#             #     Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])


#     Variables2(Cadena)


#     if contador % 1000==0: #(ChainSize//100)==0:
#         print("Progreso=",contador/ChainSize)
#         with open('dataMexico.pkl', 'wb') as f:
#             pickle.dump(Cadena, f)

        
#         # GraficaCadena(33)
#         # CadenaMarginalBeta(0)
#         # GraficaCadena(0)

#         # plt.plot()
#         # for i in range(L):
            
#         #     plt.scatter(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1])
#         # plt.show()

# with open('dataMexico.pkl', 'wb') as f:
#     pickle.dump(Cadena, f)
# # GraficaCadena(31)



# ########################################################################
# ########################################################################
# ########################################################################
# ########################################################################



with open("./dataMexico.pkl", 'rb') as f:
    Cadena = pickle.load(f)



Variables2(Cadena)


len(Cadena)

############################################################################
############################################################################
############################################################################
############################################################################


# paralell= get_context("fork").Pool(cpu_count())
# logPost = paralell.map(PostParalel, np.arange(len(Cadena)))
# paralell.close()     
# paralell.join()   
# with open('PosteriorMexico.pkl', 'wb') as f:
#     pickle.dump(logPost, f)

############################################################################
############################################################################
############################################################################
############################################################################



# Load
with open("./PosteriorMexico.pkl", 'rb') as f:
    logPost = pickle.load(f)



############################################################################
############################################################################
############################################################################
############################################################################




# with open("./dataMexico.pkl", 'rb') as f:
#     Cadena = pickle.load(f)
# f.close()

# Labels = []
# for i in range(len(Cadena)):
#     Labels.append(np.copy(Cadena[i][-2]))

# Cadena.clear()

# import gc
# gc.collect()



# MAP=np.argmax(logPost)
# # len(np.unique(Cadena[MAP][3]))

# labels=np.arange(0,L) #np.unique(Cadena[-1][-2])
# # ordena1=Cadena[MAP][-2]
# ordena1=Labels[MAP]

# # i=10000
# maps=[]
# for i in range(len(Labels)): #Labels o Cadena
#     # ordena2=Cadena[i][-2]
#     ordena2=Labels[i][-2]
#     Costo=np.zeros((L,L))
    
#     for i in range(L):
#         for j in range(L):
#             # try:
#             Costo[i,j]=np.sum(ordena1[ordena2==i] != j)
#             # except:
#             #     Costo[i,j]=0
    
    
#     xAxis,yAxis=linear_sum_assignment(Costo)
    
#     mapping = {labels[row]: labels[col] for row, col in zip(xAxis, yAxis)}
#     maps.append(mapping)
    
#     # print(len(maps)/len(Cadena))
#     print(len(maps)/len(Labels))
    
    
# with open('mapsMexico.pkl', 'wb') as f:
#     pickle.dump(maps, f)









############################################################################


labels=np.arange(0,L) #np.unique(Cadena[-1][-2])
# i=10000
maps2=[]
for i in range(len(Cadena)):
    ordena2=np.ones(L)
    for l in range(L):
        ordena2[l]=Cadena[i][-1][l][0][0]
    xAxis=np.argsort(np.arange(L))
    yAxis=np.argsort(ordena2)
        
    
    mapping = {labels[row]: labels[col] for row, col in zip(xAxis, yAxis)}
    maps2.append(mapping)
    # inverse_mapping = {labels[col]: labels[row] for row, col in zip(row_ind, col_ind)}
    print(len(maps2)/len(Cadena))
    



############################################################################
############################################################################
############################################################################
############################################################################
with open("./mapsMexico.pkl", 'rb') as f:
    maps = pickle.load(f)




for i in range(L):
    CadenaMarginalBetaLabel(i,maps,Cadena)
    # CadenaMarginalBetaLabel(i,maps2,Cadena)
    CadenaMarginalBeta(i)




# len(Cadena)
# len(logPost)



plt.plot(logPost)
plt.show()

Burnin=1000
# integrated_time(logPost)
plt.plot(logPost[Burnin:])
plt.show()



lag=4000
plt.plot(logPost[Burnin::lag])
plt.show()


print(len(np.arange(Burnin,len(Cadena),lag)))
BurninChain=np.ones(len(np.arange(Burnin,len(Cadena))))
j=3 #Revisar uno especifico
for j in range(pT):
    for i in np.arange(Burnin,len(Cadena)):
        BurninChain[i-Burnin]=Cadena[i][0][j]#alpha
        # BurninChain[i-Burnin]=Cadena[i][2][j]#gammaParam
    plt.plot(BurninChain[Burnin::lag])    
    plt.show()
    
    plot_acf(BurninChain[Burnin::lag],lags=10)
    plt.show()

    

# integrated_time(BurninChain)


plt.plot(BurninChain[Burnin::lag])
plt.show()


plt.plot(BurninChain[:])
plt.show()


# j=L-2
for j in range(L):
    for i in np.arange(Burnin,len(Cadena)):
        BurninChain[i-Burnin]=Cadena[i][1][0][maps2[i][j]]#beta
        # BurninChain[i-Burnin]=Cadena[i][4][maps[i][j]][0][0]#Phi
    plot_acf(BurninChain[Burnin::lag],lags=10)
    plt.show()
    plt.plot(BurninChain[Burnin::lag])    
    plt.show()
    
    
    
###########

######################################



GraficaCadena(0)
GraficaCadena(1)
GraficaCadena(2)
# GraficaCadena(31)
# GraficaCadena(32)
# GraficaCadena(33)

##################

MuestraEfectiva=np.arange(len(Cadena))[Burnin::lag]
print(len(MuestraEfectiva))
    



############################################################
for pTGrupo in range(pT):
    for i in MuestraEfectiva:
        j=0
        # plt.scatter(np.arange(L),Cadena[i][1][pTGrupo], alpha=0.2)
        
        plt.scatter(np.arange(L),Cadena[i][1][pTGrupo][list(maps2[i].values())], alpha=0.2)   ########Esto reetiqueta
        plt.axhline(1/L)
        # plt.plot(np.arange(L),Cadena[i][1][pTGrupo], alpha=0.1)
        j+=1
    plt.title(str(pTGrupo))
    plt.show()
    

############################################################



for i in range(pT):
    print(i)
    PDF0=DensPost(i,MuestraEfectiva,Cadena)
    if i==0:
        PDF=[PDF0]
    else :
        PDF.append(PDF0)


for i in range(pT):
    print(i)
    mu0,var0=MeanVar(PDF[i])
    CV0=np.sqrt(var0)/mu0
    if i==0:
        CV=[CV0]
    else :
        CV.append(CV0)




# for i in range(pT):
#     mu0,var0=MeanVar(PDF[i])
    # GrafPost(0,mu0,var0,np.max(PDF),CutPoint,i,CV)


# for i in range(pT):
#     mu0,var0=MeanVar(PDF[i])
#     GrafPost(0,mu0,var0,np.max(PDF),CutPoint,i)






import numpy as np
import matplotlib.pyplot as plt


Variables3((mindate,CutTime))
PostMalla(PDF,CV,0,1)
PostMalla(PDF,CV,1,1)


  
############################################################
############################################################
############################################################
############################################################
############################################################


cmap2 = plt.cm.get_cmap('viridis', pT)


columns=pT//2
fig, axes = plt.subplots(2, columns, figsize=(16, 8), constrained_layout=True)
gammahist=np.zeros(len(np.arange(Burnin,len(Cadena),lag)))

for j in range(pT):
    contador=0
    for i in np.arange(Burnin,len(Cadena),lag):
        # gammahist[contador]=Cadena[i][2][maps[i][j]]#gammaParam
        gammahist[contador]=Cadena[i][2][j]#gammaParam
        contador+=1
    row, col = divmod(j, columns)
    ax = axes[row, col]
    ax.set_xlim(0,0.06)
    ax.hist(gammahist, alpha=0.4, color=cmap2(j),density=True)
    ax.set_title(r'$\gamma_{%d}$' % (j+1))
    
    # if j<pT/2:
    #     ax.axvline(50, color='blue', linestyle='--')
    # else:
    #     ax.axvline(100, color='blue', linestyle='--')


    
    if row==1:
        ax.set_xlabel('Value')
    if col==0:
        ax.set_ylabel('Frequency')


plt.show()
len(gammahist)







# columns=pT//2
# fig, axes = plt.subplots(2, columns, figsize=(16, 8), constrained_layout=True)
# gammahist=np.zeros(len(np.arange(Burnin,len(Cadena),lag)))


# for j in range(pT):
#     contador=0
#     for i in np.arange(Burnin,len(Cadena),lag):
#         # gammahist[contador]=Cadena[i][2][maps[i][j]]#gammaParam
#         gammahist[contador]=Cadena[i][2][j]#gammaParam
#         contador+=1
#     row, col = divmod(j, columns)
#     ax = axes[row, col]
#     ax.hist(gammahist*CutPoint[1], alpha=0.4, color=cmap2(j),density=True)

#     ax.set_title(r'$E[N_%d]$' % (j+1))

    
#     # if j<pT/2:
#     #     ax.axvline(50, color='blue', linestyle='--')
#     # else:
#     #     ax.axvline(100, color='blue', linestyle='--')


    
#     if row==1:
#         ax.set_xlabel('Value')
#     if col==0:
#         ax.set_ylabel('Frequency')


# plt.show()




############################################################Clustering
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################





pi=np.zeros((len(SimPoisson),len(SimPoisson)))
contador=0
for m in np.arange(Burnin,len(Cadena),lag):
    for i in range(len(SimPoisson)):
        for j in range(i):
            valor=int(maps[contador][Cadena[m][3][i]]==maps[contador][Cadena[m][3][j]])
            pi[i,j]+=valor
            pi[j,i]+=valor
    print(m/len(Cadena))
    contador+=1
pi=pi/len(np.arange(Burnin,len(Cadena),lag))
plt.imshow(pi)
plt.show()



Clusters=np.zeros(L)

for l in np.arange(1,L+1):
    n_clusters = l  # Specify the desired number of clusters
    from sklearn.cluster import SpectralClustering
    
    # Initialize SpectralClustering with 'precomputed' affinity
    spectral_model = SpectralClustering(n_clusters=n_clusters,
                                        affinity='precomputed',
                                        random_state=1) 
    
    labelsOpt = spectral_model.fit_predict(pi)
    
    
    piaux=np.zeros((len(SimPoisson),len(SimPoisson)))
    for i in range(len(SimPoisson)):
        for j in range(len(SimPoisson)):
            piaux[i,j]+=int(labelsOpt[i]==labelsOpt[j])
        
    Clusters[l-1]=np.sum((pi-piaux)**2)


plt.plot(np.arange(1,L+1),Clusters)
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.show()





n_clusters = 8  # Specify the desired number of clusters
cmap3 = plt.cm.get_cmap('plasma', n_clusters)
from sklearn.cluster import SpectralClustering

# Initialize SpectralClustering with 'precomputed' affinity
spectral_model = SpectralClustering(n_clusters=n_clusters,
                                    affinity='precomputed',
                                    random_state=1) 

labelsOpt = spectral_model.fit_predict(pi)


fig, ax = plt.subplots(1,1, figsize=(12, 6), constrained_layout=True)
ax.set_xlim(xlim,Xlim)    
ax.set_ylim(ylim,Ylim)
fig.gca().set_facecolor('lightblue')
shape=gpd.read_file("./dest23gw/dest23gw.shp")
shape.plot(ax=ax, edgecolor='lightgray', facecolor='yellowgreen', zorder=0)     


for i in range(len(CutTime)-1):
    EarthquakesAux=Earthquakes.loc[ (Diferencias>=CutTime[i]) & (Diferencias<=CutTime[i+1])]
    labelsOptAux=labelsOpt[(Diferencias>=CutTime[i]) & (Diferencias<=CutTime[i+1])]
    if i==0:
        plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"], c=cmap3(labelsOptAux),label="p="+str(i+1), marker="o")
    if i==1:
        plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"], c=cmap3(labelsOptAux),label="p="+str(i+1), marker="^")
    if i==2:
        plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"], c=cmap3(labelsOptAux),label="p="+str(i+1), marker="s")
    if i==3:
        plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"], c=cmap3(labelsOptAux),label="p="+str(i+1), marker="P")
    


plt.plot(Trench["x"],Trench["y"],color="hotpink")    
plt.xlabel("Longitude")
plt.ylabel("Latitude")
leg = plt.legend()

for handle, text in zip(leg.legend_handles, leg.get_texts()):
    text.set_color("black")  # set text
    handle.set_facecolor("black")  # for scatter
    handle.set_edgecolor("black")
        
plt.show()     









############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################




from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

maxclust=20
Lap=laplacian(pi, normed=True)
evals, _ = eigh(Lap)
gaps = np.diff(evals[:maxclust])
plt.plot(evals[:maxclust], marker='o')
plt.grid()
plt.show()
np.argmax(gaps) + 1




############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################


cmap3 = plt.cm.get_cmap('YlOrRd', pT)



Mapa1 = './2001.geojson'
Mapa2 = './2006.geojson'
Mapa3 = './2009.geojson'
Mapa4 = './2014.geojson'
df1 = gpd.read_file(Mapa1)
df2 = gpd.read_file(Mapa2)
df3 = gpd.read_file(Mapa3)
df4 = gpd.read_file(Mapa4)


p=gpd.GeoSeries(df1.geometry)
p2=gpd.GeoSeries(df2.geometry)
p3=gpd.GeoSeries(df3.geometry)
p4=gpd.GeoSeries(df4.geometry)



fig, ax = plt.subplots()

fig.gca().set_facecolor('lightblue')
shape=gpd.read_file("./dest23gw/dest23gw.shp")
shape.plot(ax=ax, edgecolor='lightgray', facecolor='yellowgreen', zorder=0)     
for i in range(len(CutTime)-1):
    EarthquakesAux=Earthquakes.loc[ (Diferencias>=CutTime[i]) & (Diferencias<=CutTime[i+1])]
    # print(len(EarthquakesAux))
    # xlim,Xlim=-106,-96
    # ylim,Ylim=14.5,20
    plt.xlim(xlim,Xlim)    
    plt.ylim(ylim,Ylim)
    npartX=20
    npartY=20
    DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
    DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
    # shape=shp.Reader("./Mexico.shp") #http://geoportal.conabio.gob.mx/metadatos/doc/html/dest_2010gw.html
    # for shape in shape.shapeRecords():
    #     x = [i[0] for i in shape.shape.points[:]]
    #     y = [i[1] for i in shape.shape.points[:]]
    #     plt.plot(x,y,color="black")    

    plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"],label="p="+str(i+1),color=cmap3(i),alpha=0.7)
    plt.legend(loc="center left" )
    
    plt.xlim(DomX[0],DomX[-1])
    plt.ylim(DomY[0],DomY[-1])
    # plt.title("Interval:"+str(round(CutTime[i],2))+"-"+str(round(CutTime[i+1],2)))
    # plt.title("Interval:"+str((mindate+timedelta(days=CutTime[i])))+"-"+str((mindate+timedelta(days=CutTime[i+1]))))
    # plt.show()
plt.plot(Trench["x"],Trench["y"],color="hotpink")    
p=gpd.GeoSeries(df1.geometry[4])
p2=gpd.GeoSeries(df2.geometry[3])
p3=gpd.GeoSeries(df3.geometry[3])
p4=gpd.GeoSeries(df4.geometry)
p.plot(ax=ax,alpha=0.8,facecolor='none',edgecolor="blue")
p2.plot(ax=ax,alpha=0.8,edgecolor="orange",facecolor='none')
p3.plot(ax=ax,alpha=0.8,edgecolor="crimson",facecolor='none')
p4.plot(ax=ax,alpha=0.8,edgecolor="black",facecolor='none')    


gdf = gpd.read_file("FFM2003.geojson")  # Replace with your file path
# gdf.plot(ax=ax,column='slip',edgecolor='black', cmap='hot',figsize=(10, 10))

centroids = gdf.geometry.centroid
x = centroids.x.values
y = centroids.y.values
z = gdf['slip'].values  # Replace with your actual data column
# Create a grid for interpolation
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')
contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1),alpha=0.7)
# ax.clabel(contours, inline=True, fontsize=8)
# plt.show()


# 
gdf2 = gpd.read_file("FFM2012.geojson")  # Replace with your file path
centroids = gdf2.geometry.centroid
x = centroids.x.values
y = centroids.y.values
z = gdf2['slip'].values  # Replace with your actual data column
# Create a grid for interpolation
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')
contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1),alpha=0.7)
# 
gdf3 = gpd.read_file("FFM2014.geojson")  # Replace with your file path
centroids = gdf3.geometry.centroid
x = centroids.x.values
y = centroids.y.values
z = gdf3['slip'].values  # Replace with your actual data column
# Create a grid for interpolation
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')
contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1),alpha=0.7)
plt.ylabel("Latitude")
plt.xlabel("Longitude")    
plt.show()    
###################





































