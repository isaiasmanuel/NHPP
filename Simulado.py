#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:50:07 2025

@author: isaias
"""

import os 

os.chdir('/Users/isaias/Desktop/Poisson_Earthquakes/')

from Funciones import *



####################





############################## Generacion de la muestra sintetica

#########
# x=0
# y=0
# t=0
# ###WSImulacion usando thining
# Tmax=10
# def cambioT(t):
#     return 1/(1+np.exp(-(t-Tmax/2)))
# # plt.plot(np.arange(0,Tmax,0.01),cambioT(np.arange(0,Tmax,0.01)))
# # plt.show()
# def cambiogamma(t):
#     # return (1/(1+np.exp(-50*(t-Tmax/2)))+0.5)*10
#     return (t<Tmax/2)*50+(t>=Tmax/2)*100
# # plt.plot(np.arange(0,Tmax,0.01),cambiogamma(np.arange(0,Tmax,0.01)))
# # plt.show()
# def intensity(X):
#     x,y,t=X
#     return cambiogamma(t)*(cambioT(t)*(2/3*multivariate_normal.pdf((x,y), mean=(0,0),cov=1)+1/3*multivariate_normal.pdf((x,y), mean=(2,2),cov=1))+(1-cambioT(t))*(1/3*multivariate_normal.pdf((x,y), mean=(4.,6.),cov=1)+2/3*multivariate_normal.pdf((x,y), mean=(6,2),cov=1)))
# TimeCuts=6
# for t in np.arange(0,Tmax+Tmax/TimeCuts,Tmax/TimeCuts):
#     x = np.linspace(-5,10,20)
#     y = np.linspace(-5,10,20)
#     Xg,Yg = np.meshgrid(x,y)
#     pdf0 = np.zeros(Xg.shape)
#     for i in range(Xg.shape[0]):
#         for j in range(Xg.shape[1]):
#             Obs=np.array((Xg[i,j], Yg[i,j],t))
#             pdf0[i,j] = intensity(Obs)    
#     plt.plot()
#     contour=plt.contourf(Xg, Yg, pdf0, cmap='viridis',levels=np.linspace(0, 10, 10000),vmax=10)
#     plt.colorbar(contour)
#     plt.title( "t="+str(np.round(t,2)))
#     plt.show()

    
# np.max(pdf0)    

# Const=[[-5, 5], #x
#   [-5, 5],#y
#   [0.0, Tmax]]#t


# Opt=sp.optimize.minimize(lambda theta: -intensity(theta),(0.5,0.5,Tmax/2+1), bounds=Const, method= "L-BFGS-B")
# # Opt=sp.optimize.minimize(lambda theta: -intensity(theta),(0.5,0.5,Tmax/2+1), bounds=Const, method= "Nelder-Mead")
# intensmax=intensity(Opt["x"])

# Eventos=poisson.rvs(intensity(Opt["x"])*15*15*Tmax,size=1)[0]
# # plt.hist(poisson.rvs(intensity(Opt["x"])*15*15*Tmax,size=1000)) #|D|*T donde |D| es el area donde simulo y T la longitud en tiempo
# # plt.show()

# Candidatos=np.vstack((np.random.uniform(-5,10,Eventos),np.random.uniform(-5,10,Eventos),np.sort(np.random.uniform(0,Tmax,Eventos)))).T

# Aceptados=np.zeros(Eventos)
# for i in range(Eventos):
#     paccept=intensity(Candidatos[i,:])/intensmax
#     Aceptados[i]=np.random.choice(2,p=(1-paccept,paccept))

# len(Aceptados)
# SimPoisson=Candidatos[Aceptados==1]
# np.sum(SimPoisson[:,2]<2.5)/2.5
# np.sum(SimPoisson[:,2]>7.5)/2.5
# # r=0
# # TimeCuts=9
# # for t in np.arange(0,Tmax+Tmax/TimeCuts,Tmax/TimeCuts):
# #     x = np.linspace(-5,10,20)
# #     y = np.linspace(-5,10,20)
# #     Xg,Yg = np.meshgrid(x,y)
# #     pdf = np.zeros(Xg.shape)
# #     for i in range(Xg.shape[0]):
# #         for j in range(Xg.shape[1]):
# #             Obs=np.array((Xg[i,j], Yg[i,j],t))
# #             pdf[i,j] = intensity(Obs)
# #     plt.plot()
# #     plt.contourf(Xg, Yg, pdf, cmap='viridis',levels=1000)
# #     plt.title(t)
# #     # plt.colorbar()
# #     Elegidos=(SimPoisson[:,2]<np.arange(0,Tmax+Tmax/TimeCuts,Tmax/TimeCuts)[r+1]) & (SimPoisson[:,2]>np.arange(0,Tmax+Tmax/TimeCuts,Tmax/TimeCuts)[r])
# #     plt.scatter(SimPoisson[:,0][Elegidos], SimPoisson[:,1][Elegidos],alpha=1, color="red")
# #     plt.show()
# #     r+=1

##### np.save("Muestra", SimPoisson)



SimPoisson=np.load("Muestra.npy")
Tmax=10
xlim,Xlim=-5,10
ylim,Ylim=-5,10

#######Hyperparameters
#Particiones T
pT=8
CutPoint=np.append((np.arange(0,pT)/pT)*Tmax,Tmax)
#Aproximacion finita
ClustTiempo=1
L=pT*ClustTiempo

cmap = plt.cm.get_cmap('viridis', L)

#NIW
nu=3 #Chackrabarti usa nu=2, lam=0.01,Sprior Identity Mu=0,0
Sprior=np.diag((1,1)) #Parece funcionar nu=10, Sprior=10I, mu=1,1, lam=2
mu=np.array((1,1))
lam=0.1
#alpha0
alpha0=1
#
pgamma=70
k=0.1
GammaPrior=gamma(a=pgamma*k,scale=1/k)
GammaPrior.rvs()
IW=invwishart(df=nu,scale=Sprior)
Ssim=IW.rvs()
Gaus=multivariate_normal(mean=mu, cov=Ssim/lam)
muSim=Gaus.rvs()
print(muSim,Ssim)

SimPoisson=np.hstack((SimPoisson,np.zeros((len(SimPoisson),1))))
for i in range(pT):    
    SimPoisson[(CutPoint[i]<SimPoisson[:,2])&(SimPoisson[:,2]<CutPoint[i+1]),3]=i

SimPoisson[:,3]
############################
Alpha=np.ones(pT)
Beta=np.ones((pT,L))/L
Z=np.random.choice(np.arange(L),size=len(SimPoisson))
gammaParam=GammaPrior.rvs(pT)

for i in range(L):
    IW=invwishart(df=nu,scale=Sprior)
    Ssim=IW.rvs()
    Gaus=multivariate_normal(mean=mu, cov=Ssim/lam)
    muSim=np.random.uniform(-5,10,size=2) #Gaus.rvs()
    
    
    if i==0:
        Phi=[[muSim,Ssim]]
    else :
        Phi.append([muSim,Ssim])




Cadena=[[Alpha,Beta,gammaParam,Z,Phi]]

print(len(SimPoisson))
print(np.sum(L*pT*6+pT+pT*L+L))


ChainSize=5_000_00

np.random.seed(1)

Variables((Tmax,pT,xlim,Xlim,ylim,Ylim,SimPoisson,L,alpha0,IW,Gaus,GammaPrior))




# Actualizaciones=np.zeros(0)


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
#         with open('data.pkl', 'wb') as f:
#             pickle.dump(Cadena, f)
#         # GraficaCadena(31)
        
#         GraficaCadena(33)
#         CadenaMarginalBeta(0)
#         GraficaCadena(0)

#         # plt.plot()
#         # for i in range(L):
            
#         #     plt.scatter(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1])
#         # plt.show()

# with open('data.pkl', 'wb') as f:
#     pickle.dump(Cadena, f)
# # GraficaCadena(31)



# ########################################################################
# ########################################################################
# ########################################################################
# ########################################################################



with open("./data.pkl", 'rb') as f:
    Cadena = pickle.load(f)

Variables2(Cadena)


len(Cadena)

############################################################################
############################################################################
############################################################################
############################################################################


paralell= get_context("fork").Pool(cpu_count())
logPost = paralell.map(PostParalel, np.arange(len(Cadena)))
paralell.close()     
paralell.join()   
with open('Posterior.pkl', 'wb') as f:
    pickle.dump(logPost, f)

############################################################################
############################################################################
############################################################################
############################################################################



# Load
with open("./Posterior.pkl", 'rb') as f:
    logPost = pickle.load(f)



############################################################################
############################################################################
############################################################################
############################################################################ Si no funciona usar el caso de Mexico


MAP=np.argmax(logPost)
labels=np.unique(Cadena[-1][-2])
ordena1=Cadena[MAP][-2]
# i=10000
maps=[]
for i in range(len(Cadena)):
    ordena2=Cadena[i][-2]
    Costo=np.zeros((len(labels),len(labels)))
    
    for i in labels:
        for j in labels:
            Costo[i,j]=np.sum(ordena1[ordena2==i] != j)
    
    
    xAxis,yAxis=linear_sum_assignment(Costo)
    
    mapping = {labels[row]: labels[col] for row, col in zip(xAxis, yAxis)}
    maps.append(mapping)
    # inverse_mapping = {labels[col]: labels[row] for row, col in zip(row_ind, col_ind)}
    print(len(maps)/len(Cadena))
    
with open('maps.pkl', 'wb') as f:
    pickle.dump(maps, f)


############################################################################
############################################################################
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
with open("./maps.pkl", 'rb') as f:
    maps = pickle.load(f)



############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################



for i in range(L):
    CadenaMarginalBetaLabel(i,maps2,Cadena)
    CadenaMarginalBeta(i)







plt.plot(logPost)
plt.show()

Burnin=10000
# integrated_time(logPost)
plt.plot(logPost[Burnin:])
plt.show()


lag=4000
print(len(np.arange(Burnin,len(Cadena),lag)))
BurninChain=np.ones(len(np.arange(Burnin,len(Cadena))))
j=3 #Revisar uno especifico
for j in range(pT):
    for i in np.arange(Burnin,len(Cadena)):
        BurninChain[i-Burnin]=Cadena[i][0][j]#alpha
        # BurninChain[i-Burnin]=Cadena[i][2][j]#gammaParam
        
    plot_acf(BurninChain[Burnin::lag],lags=10)
    plt.show()
    

    
integrated_time(BurninChain)
plt.plot(BurninChain[Burnin::lag])
plt.show()

plt.plot(BurninChain[:])
plt.show()


for j in range(pT):
    for i in np.arange(Burnin,len(Cadena)):
        # BurninChain[i-Burnin]=Cadena[i][1][0][maps[i][j]]#beta
        BurninChain[i-Burnin]=Cadena[i][4][maps2[i][j]][0][0]#Phi
    plot_acf(BurninChain[Burnin::lag],lags=10)
    plt.show()
    
    plt.plot(BurninChain[Burnin::lag])
    plt.show()
    



############################################################



GraficaCadena(0)
GraficaCadena(1)
GraficaCadena(2)
# GraficaCadena(31)
# GraficaCadena(32)
# GraficaCadena(33)

##################

MuestraEfectiva=np.arange(len(Cadena))[Burnin::lag]
print(len(MuestraEfectiva))
    

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




# for i in range(L):
#     mu0,var0=MeanVar(PDF[i])
#     GrafPost(3,mu0,var0,np.max(PDF),CutPoint,i)


# for i in range(L):
#     mu0,var0=MeanVar(PDF[i])
#     GrafPost(0,mu0,var0,np.max(PDF),CutPoint,i)






import numpy as np
import matplotlib.pyplot as plt



PostMalla(PDF,CV,0)
PostMalla(PDF,CV,1)
PostMalla(PDF,CV,2)

############################################################
############################################################
############################################################
############################################################
############################################################



cmap2 = plt.cm.get_cmap('viridis', pT)



fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
gammahist=np.zeros(len(np.arange(Burnin,len(Cadena),lag)))

for j in range(pT):
    contador=0
    for i in np.arange(Burnin,len(Cadena),lag):
        # gammahist[contador]=Cadena[i][2][maps[i][j]]#gammaParam
        gammahist[contador]=Cadena[i][2][j]#gammaParam
        contador+=1
    row, col = divmod(j, 4)
    ax = axes[row, col]
    ax.hist(gammahist, alpha=0.4, color=cmap2(j),density=True)
    ax.set_title(r'$\gamma_{%d}$' % (j+1))
    
    if j<pT/2:
        ax.axvline(50, color='blue', linestyle='--')
        ax.set_xlim(30,130)
    else:
        ax.axvline(100, color='blue', linestyle='--')
        ax.set_xlim(30,130)

    
    if row==1:
        ax.set_xlabel('Value')
    if col==0:
        ax.set_ylabel('Frequency')


plt.show()




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
            valor=int(maps2[contador][Cadena[m][3][i]]==maps2[contador][Cadena[m][3][j]])
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


from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)





n_clusters = 4  # Specify the desired number of clusters
from sklearn.cluster import SpectralClustering

# Initialize SpectralClustering with 'precomputed' affinity
spectral_model = SpectralClustering(n_clusters=n_clusters,
                                    affinity='precomputed',
                                    random_state=1) 

labelsOpt = spectral_model.fit_predict(pi)

plt.scatter(SimPoisson[:,0],SimPoisson[:,1], c=labelsOpt)
plt.scatter((0,2,6,4),(0,2,2,6), marker='x',color="red")
plt.xlabel("x")
plt.ylabel("y")
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
























