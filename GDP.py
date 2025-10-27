#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:57:37 2025

@author: isaias
"""


############
from scipy.stats import invwishart,multivariate_normal,gamma,dirichlet,poisson
import numpy as np
import mpmath as mp
import scipy as sp
import matplotlib.pyplot as plt
import copy
from emcee.autocorr import integrated_time
from statsmodels.graphics.tsaplots import plot_acf
import pickle
from multiprocessing import Pool, cpu_count, get_context, Process

###########

np.random.seed(1)

def NormalZ(i):
    Zprob=np.zeros(L)
    Zprob=mp.matrix(Zprob)
    for lindex in range(L):
        Zprob[lindex]=Cadena[-1][1][int(SimPoisson[i,3])][lindex]*multivariate_normal.pdf(SimPoisson[i,[0,1]], mean=Cadena[-1][4][lindex][0],cov=Cadena[-1][4][lindex][1])    
    dens=Zprob[:]/mp.fsum(Zprob[:])
    return np.random.choice(L,p=dens)


def Post(X):
    [Alpha,Beta,gammaParam,Z,Phi]=X
    p1=(alpha0-1)*np.log(Alpha[0])-Alpha[0]
    p2=(Alpha[0]/L-1)*np.sum(np.log(Beta[0]))+mp.log(BetaInv( np.repeat(Alpha[0]/L, L)))    
    p3=mp.fsum((Alpha[:-1]-1)*np.log(Alpha[1:])-Alpha[1:]-GammaVectorized(Alpha[:-1]))
    p4=0
    for i in range(pT-1):
        p4+=mp.fsum(Alpha[i+1]*Beta[i]*np.log(Beta[i+1])+BetaInv(Alpha[i+1]*Beta[i]))
    p5=0    
    for i in range(pT):
        p5+=IW.logpdf(Phi[i][1])+Gaus.logpdf(Phi[i][0])
    p6=0
    for i in range(pT):
        p6+=np.sum(np.log(Beta[i][Z[SimPoisson[:,-1]==i]]))
    p7=np.sum(GammaPrior.logpdf(gammaParam))
    p8=0        
    for i in range(pT):
        p8=np.sum(SimPoisson[:,-1]==i)*np.log(gammaParam[i])- gammaParam[i]*Tmax/pT
        contador=0
        for j in Z[SimPoisson[:,-1]==i]:
            p8+=multivariate_normal.logpdf(SimPoisson[SimPoisson[:,-1]==i][contador][:2], mean=Phi[j][0],cov=Phi[j][1])
            contador+=1
    return p1+p2+p3+p4+p5+p6+p7+p8
    



def DensPost(SelectedInterval):
    x = np.linspace(-5,10,50)
    y = np.linspace(-5,10,50)
    Xg,Yg = np.meshgrid(x,y)
    PDF=np.zeros((len(MuestraEfectiva),len(x),len(y)))
    # for SelectedInterval in np.arange(pT):
    w=0
    for r in MuestraEfectiva:
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Obs=np.array((Xg[i,j], Yg[i,j]))
                for s in range(L):
                    # gammadens=np.exp(-Cadena[IATElegido][2][SelectedInterval] * (CutPoint[TActualizar+1]-CutPoint[TActualizar]))*(Cadena[IATElegido][2][SelectedInterval])**np.sum(SimPoisson[:,3]==SelectedInterval)
                    PDF[w,i,j] = PDF[w,i,j]+Cadena[IATElegido][2][SelectedInterval]*Cadena[r][1][SelectedInterval][s]*multivariate_normal.pdf(Obs,mean=Cadena[r][4][s][0], cov=Cadena[r][4][s][1])
        w+=1
 
    return PDF



def CadenaMarginalBeta(i):
        # plt.plot(Cadena[IATElegido::IATElegido,i])
        graf=np.zeros(len(Cadena))
        for j in range(len(Cadena)):
            graf[j]=Cadena[j][1][0][i]     
        plt.plot(graf)
        plt.title(str(i))
        plt.show()


def Graf1Beta(ax, Lindex):
    GammaSim=np.zeros((len(Cadena),L))
    for i in range(len(Cadena)):
        GammaSim[i,:]=Cadena[i][1][Lindex]
    for i in range(L):
        ax.plot(GammaSim[:,i],label=(str(i)))
        ax.set_title(Lindex)

def GraficaBeta():
    fig, axs = plt.subplots(4, 2)
    axs = axs.flatten()
    for i in range(pT):
        Graf1Beta(axs[i],np.arange(pT)[i])
        if i==0:
            fig.legend()
    plt.show()

def Graf1mu(ax, Lindex):
    GammaSim=np.zeros((len(Cadena),2))
    for i in range(len(Cadena)):
        GammaSim[i,0]=Cadena[i][4][Lindex][0][0]
        GammaSim[i,1]=Cadena[i][4][Lindex][0][1]
    for i in range(2):
        ax.plot(GammaSim[:,i],label=str(i))
    # ax.legend(loc=2)
    # ax.plot(GammaSim[:,i])
    ax.set_title(Lindex)

    # plt.title(r'$ \gamma$')
    # plt.show()
    


def Grafmu():
    fig, axs = plt.subplots(pT, ClustTiempo)
    axs = axs.flatten()
    for i in range(L):
        Graf1mu(axs[i],np.arange(L)[i])
    plt.show()

def GraficaCadena(Gra):
    if Gra==0 :
        GammaSim=np.zeros((len(Cadena),pT))
        for i in range(len(Cadena)):
            GammaSim[i,:]=Cadena[i][0]
        for i in range(pT):
            plt.plot(GammaSim[:,i],label=str(i),alpha=0.5)
        plt.legend(loc=2)
        plt.title(r'$ \alpha$')
        plt.show()
    
    if Gra==1:
        GraficaBeta()
        
    if Gra==2:
        fig, axs = plt.subplots(4, 2)
        axs = axs.flatten()
        GammaSim=np.zeros((len(Cadena),pT))
        for i in range(len(Cadena)):
            GammaSim[i,:]=Cadena[i][2]
        for i in range(pT):
            axs[i].plot(GammaSim[:,i],label=str(i),alpha=1)
        # axs.legend(loc=2)
        # plt.title(r'$ \gamma$')
        plt.show()


    if Gra==31:
        contador=0
        for t in np.arange(0,Tmax,Tmax/pT):
            x = np.linspace(-5,10,20)
            y = np.linspace(-5,10,20)
            Xg,Yg = np.meshgrid(x,y)
            pdf = np.zeros(Xg.shape)
            for i in range(Xg.shape[0]):
                for j in range(Xg.shape[1]):
                    Obs=np.array((Xg[i,j], Yg[i,j],t+Tmax/pT))
                    pdf[i,j] = intensity(Obs)

            plt.plot()
            plt.contourf(Xg, Yg, pdf, cmap='viridis',levels=100)
            plt.title(t+Tmax/pT)
            # plt.colorbar()
            # plt.show()
            alphaGraf=np.copy(Cadena[-1][1][contador])
            if len(np.unique(alphaGraf))==1 :
                alphaGraf=alphaGraf/alphaGraf
            else :
                alphaGraf=(alphaGraf-np.min(alphaGraf))/(np.max(alphaGraf)-np.min(alphaGraf))

            for i in range(L):
                plt.scatter(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1],alpha=alphaGraf[i])
                plt.text(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1],str(i))
            plt.show()
            contador+=1



    if Gra==33:
        contador=0
        for t in np.arange(0,Tmax,Tmax/pT):
            plt.plot()
            plt.xlim(-5,10)
            plt.ylim(-5,10)
            plt.scatter(SimPoisson[SimPoisson[:,3]==contador,0],SimPoisson[SimPoisson[:,3]==contador,1],c=cmap(Cadena[-1][-2][SimPoisson[:,3]==contador]))
            plt.title(t+Tmax/pT)
            # plt.colorbar()
            # plt.show()
            alphaGraf=np.copy(Cadena[-1][1][contador])
            if len(np.unique(alphaGraf))==1 :
                alphaGraf=alphaGraf/alphaGraf
            else :
                alphaGraf=(alphaGraf-np.min(alphaGraf))/(np.max(alphaGraf)-np.min(alphaGraf))

            for i in range(L):
                plt.scatter(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1],alpha=alphaGraf[i],color=cmap(i),marker='s',label=i)
                plt.text(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1],str(i))
            plt.legend(loc=1)
            plt.show()
            contador+=1

    if Gra==32:
        Grafmu()


#########
x=0
y=0
t=0
###WSImulacion usando thining
Tmax=10
def cambioT(t):
    return 1/(1+np.exp(-(t-Tmax/2)))
# plt.plot(np.arange(0,Tmax,0.01),cambioT(np.arange(0,Tmax,0.01)))
# plt.show()
def cambiogamma(t):
    # return (1/(1+np.exp(-50*(t-Tmax/2)))+0.5)*10
    return (t<Tmax/2)*50+(t>=Tmax/2)*100
# plt.plot(np.arange(0,Tmax,0.01),cambiogamma(np.arange(0,Tmax,0.01)))
# plt.show()
def intensity(X):
    x,y,t=X
    return cambiogamma(t)*(cambioT(t)*(2/3*multivariate_normal.pdf((x,y), mean=(0,0),cov=1)+1/3*multivariate_normal.pdf((x,y), mean=(2,2),cov=1))+(1-cambioT(t))*(1/3*multivariate_normal.pdf((x,y), mean=(4.,6.),cov=1)+2/3*multivariate_normal.pdf((x,y), mean=(6,2),cov=1)))
TimeCuts=6
for t in np.arange(0,Tmax+Tmax/TimeCuts,Tmax/TimeCuts):
    x = np.linspace(-5,10,20)
    y = np.linspace(-5,10,20)
    Xg,Yg = np.meshgrid(x,y)
    pdf0 = np.zeros(Xg.shape)
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            Obs=np.array((Xg[i,j], Yg[i,j],t))
            pdf0[i,j] = intensity(Obs)    
    plt.plot()
    contour=plt.contourf(Xg, Yg, pdf0, cmap='viridis',levels=np.linspace(0, 10, 10000),vmax=10)
    plt.colorbar(contour)
    plt.title( "t="+str(np.round(t,2)))
    plt.show()

    
np.max(pdf0)    

Const=[[-5, 5], #x
  [-5, 5],#y
  [0.0, Tmax]]#t

############################## Generacion de la muestra sintetica

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
GammaPrior=gamma(a=pgamma)
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


#######
X=np.ones(5)##Es basura para probar Betainv

GammaVectorized=np.vectorize(mp.gamma)
def BetaInv(X):    #Es el inverso multiplicativo de la Beta
    Aux=GammaVectorized(np.hstack((X,mp.fsum(X))))
    return mp.fdiv(Aux[-1],mp.fprod(Aux[:-1]))


def a(alpha,Beta,i):
    if i==0:
        num=mp.exp(-alpha[i])*mp.power(alpha[i],(alpha0-1))*mp.power(alpha[i+1],(alpha[i])) * mp.fprod(Beta[i,:]**(alpha[i]/L))* BetaInv( np.repeat(alpha[i]/L, L))
        den=mp.gamma(alpha[i])
    elif i<(pT-1):
        num=mp.exp(-alpha[i])*mp.power(alpha[i],(alpha[i-1]-1))*mp.power(alpha[i+1],(alpha[i])) * mp.fprod(Beta[i,:]**(alpha[i]*Beta[i-1,:] ))* BetaInv( mp.mpmathify(alpha[i-1])*(Beta[i-1,:])) 
        den=mp.gamma(alpha[i])
    else :
        num=mp.exp(-alpha[i])*mp.power(alpha[i],(alpha[i-1]-1)) * mp.fprod(Beta[i,:]**(alpha[i]*Beta[i-1,:] ))* BetaInv( mp.mpmathify(alpha[i-1])*(Beta[i-1,:])) 
        den=1 #mp.gamma(alpha[i])
    return num/den

def b1(alpha,Beta,Z):
    m1l=np.zeros(L)
    for i in range(L):
        m1l[i]=np.sum(Z[SimPoisson[:,3]==0]==i)
        # m1l[i]=np.sum(Z==i)
    return mp.fprod(Beta[0,:]**(m1l+alpha[0]/L-1))*mp.fprod(Beta[1,:]**(alpha[1]*Beta[0,:]))*BetaInv(alpha[1]*Beta[0,:])

# def PhiFunc(muSim,Ssim,l):
#     mp.fprod(multivariate_normal.pdf(SimPoisson[Z==l,:2],mean=mu, cov=Ssim/lam))*multivariate_normal.pdf(muSim,mean=mu, cov=Ssim/lam)*invwishart.pdf(Ssim,df=nu,scale=Sprior)
#IW.pdf(Ssim)*multivariate_normal(mean=mu, cov=Ssim/lam)

##########################################################################MCMC


def SALT(beta):
    coord=np.random.choice(len(beta))
    aislado=np.random.choice(len(beta))
    bipropuesto=np.copy(beta)
    bi=beta[coord]
    logitbi=mp.log(mp.fdiv(bi,mp.fsum((1,-bi))))+np.random.normal(size=1,scale=0.1*bi*(1-bi) )[0]
    bipropuesto[coord]=mp.fdiv(mp.exp(logitbi),1+mp.exp(logitbi))   
    for i in range(len(bipropuesto)):
        if i!=coord and i!=aislado:
            bipropuesto[i]= (1-bipropuesto[coord]) * mp.fsum((( beta[i]/(1-beta[i]),np.random.uniform(-10**-20,10**-20))))
    bipropuesto[aislado]=1-mp.fsum(np.delete(bipropuesto, aislado))
    # print(bipropuesto,mp.fsum(bipropuesto))
    qratio=mp.fprod((bipropuesto[coord],(1-bipropuesto[coord])**(L-1)))/mp.fprod((bi,(1-bi)**(L-1)))    
    # if all(bipropuesto>0.001):
    #     return (bipropuesto,qratio)
    # else :
    #     return (beta,1)
    return (bipropuesto,qratio)
##########################################################################

Cadena=[[Alpha,Beta,gammaParam,Z,Phi]]

# print(len(SimPoisson))
# print(np.sum(L*pT*6+pT+pT*L+L))


ChainSize=1000000
np.random.seed(1)
for contador in range(ChainSize):
    
    # ActWeigh=np.array((5,20,1,1,1))
    ActWeigh=np.array((5,20,1,1,1))
    Actualizar=np.random.choice(5, p=ActWeigh/np.sum(ActWeigh))
    if Actualizar==0:
        TActualizar=np.random.choice(pT)
        # for TActualizar in range(pT):
        Alphaprop=np.copy(Cadena[-1][Actualizar])
        Alphaprop[TActualizar]=Alphaprop[TActualizar]+np.random.normal(loc=0,scale=0.1)
        if Alphaprop[TActualizar]>0 :
            rho=min(1,a(Alphaprop,Cadena[-1][1],TActualizar)/a(Cadena[-1][0],Cadena[-1][1],TActualizar))
            if np.random.choice(2, p=(1-rho,rho))==1:
                Cadena.append([Alphaprop,Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])
            else :
                Cadena.append(Cadena[-1])
        else :
            Cadena.append(Cadena[-1])


    if Actualizar==1:
        probaBetaT=np.ones(pT)
        probaBetaT[0]=10
        TActualizar=np.random.choice(pT, p=probaBetaT/np.sum(probaBetaT))
        # for TActualizar in range(pT):
        if TActualizar==0:
            betaprop,qratio=SALT(Cadena[-1][Actualizar][TActualizar,:])
            Betaprop=np.copy(Cadena[-1][Actualizar])
            Betaprop[TActualizar]=betaprop
            if all(Betaprop[TActualizar]>0):
                rho=min(1,qratio*b1(Cadena[-1][0],Betaprop,Cadena[-1][3])/b1(Cadena[-1][0],Cadena[-1][Actualizar],Cadena[-1][3]  ) )
            else :
                rho=0
            if np.random.choice(2, p=(1-rho,rho))==1:
                Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])
            else :
                Cadena.append(Cadena[-1])
        if TActualizar!=0:
            m1l=np.zeros(L)
            for i in range(L):
                m1l[i]=np.sum(Z[SimPoisson[:,3]==TActualizar]==i)
                # m1l[i]=np.sum(Z==i)
            Betaprop=np.copy(Cadena[-1][Actualizar])
            Betaprop[TActualizar]=dirichlet.rvs(m1l+Cadena[-1][0][TActualizar]*Cadena[-1][1][TActualizar-1],  size=1)
            if not all(Betaprop[TActualizar]>0):
                # Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
                Betaprop[TActualizar]=Betaprop[TActualizar]+np.finfo(float).eps
                Betaprop[TActualizar]=Betaprop[TActualizar]/np.sum(Betaprop[TActualizar])
                Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
            if all(Betaprop[TActualizar]>0):
                Cadena.append([Cadena[-1][0],Betaprop,Cadena[-1][2],Cadena[-1][3],Cadena[-1][4]])    
            else :
                Cadena.append(Cadena[-1])


    if Actualizar==2:
        # TActualizar=np.random.choice(pT)
        gamaprop=np.copy(Cadena[-1][2])
        for TActualizar in range(pT):
            gamaprop[TActualizar]=gamma.rvs( a=np.sum(SimPoisson[:,3]==TActualizar)+pgamma,scale=1/(1+(CutPoint[TActualizar+1]-CutPoint[TActualizar]))  ) 
            Cadena.append([Cadena[-1][0],Cadena[-1][1],gamaprop,Cadena[-1][3],Cadena[-1][4]])
    if Actualizar==3:
        
        ########Idea1
        # for TActualizar in range(pT):
        #     ZPropuesta=np.copy(Cadena[-1][Actualizar])
        #     for i in range(len(ZPropuesta)):
        #         Zprob=np.zeros(L)
        #         Zprob=mp.matrix(Zprob)
        #         for lindex in range(L):
        #             Zprob[lindex]=Cadena[-1][1][TActualizar][lindex]*multivariate_normal.pdf(SimPoisson[i,[0,1]], mean=Cadena[-1][4][lindex][0],cov=Cadena[-1][4][lindex][1])
        #         dens=Zprob[:]/mp.fsum(Zprob[:])
        #         ZPropuesta[i]=np.random.choice(L,p=dens)
        #     # plt.scatter(SimPoisson[ZPropuesta==0][:,0],SimPoisson[ZPropuesta==0][:,1])
        ########Idea2
        # ZPropuesta=np.copy(Cadena[-1][Actualizar])
        # for i in range(len(SimPoisson)):
        #     ZPropuesta[i]=NormalZ(i)
        ########Idea3
        paralell= get_context("fork").Pool(cpu_count())
        results = paralell.map(NormalZ, np.arange(len(SimPoisson)))
        paralell.close()        
        ZPropuesta=np.array(results)
        Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],ZPropuesta,Cadena[-1][4]])
        
    if Actualizar==4:
        # LActualizar=np.random.choice(L)
        for LActualizar in range(L):
            MuestraAux=SimPoisson[Cadena[-1][3]==LActualizar,:2]
            if len(MuestraAux)!=0:
                # plt.scatter(SimPoisson[Cadena[-1][3]==LActualizar,:2][:,0],SimPoisson[Cadena[-1][3]==LActualizar,:2][:,1])
                # len(MuestraAux)
                ybar=np.apply_along_axis(np.mean, 0,MuestraAux)
                np.reshape((ybar-mu),(2,1))@np.reshape((ybar-mu),(1,2))
                Diferencia=(MuestraAux-ybar)
                S=0
                for i in range(len(MuestraAux)):
                    S+=np.reshape(Diferencia[i],(2,1))@np.reshape(Diferencia[i],(1,2))
                naux=len(MuestraAux)
                mun=(mu*lam+naux*ybar)/(lam+naux)
                lamn=lam+naux
                nun=nu+naux
                Spost=Sprior+S+ (lam*naux)/lamn*np.reshape((ybar-mu),(2,1))@np.reshape((ybar-mu),(1,2))
                IW=invwishart(df=nun,scale=Spost)
                Ssim2=IW.rvs()
                Gaus=multivariate_normal(mean=mun, cov=Ssim2/lamn)
                muSim2=Gaus.rvs()
                # print(muSim2,Ssim2)
                PhiPropuesta=copy.deepcopy(Cadena[-1][Actualizar])
                PhiPropuesta[LActualizar][0]=np.copy(muSim2)
                # Cadena[-1][Actualizar][LActualizar][0]
                PhiPropuesta[LActualizar][1]=np.copy(Ssim2)
                Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])
            if len(MuestraAux)==0:
                IW=invwishart(df=nu,scale=Sprior)
                Ssim2=IW.rvs()
                Gaus2=multivariate_normal(mean=mu, cov=Ssim2/lam)
                muSim2=Gaus.rvs()
                PhiPropuesta=copy.deepcopy(Cadena[-1][Actualizar])
                PhiPropuesta[LActualizar][0]=np.copy(muSim2)
                # Cadena[-1][Actualizar][LActualizar][0]
                PhiPropuesta[LActualizar][1]=np.copy(Ssim2)
                Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])

                
                
            
            # else :
            #     Cadena.append([Cadena[-1][0],Cadena[-1][1],Cadena[-1][2],Cadena[-1][3],PhiPropuesta])
    if contador % 1000==0: #(ChainSize//100)==0:
        print("Progreso=",contador/ChainSize)
        with open('data.pkl', 'wb') as f:
            pickle.dump(Cadena, f)
        # GraficaCadena(31)
        GraficaCadena(33)
        CadenaMarginalBeta(0)
        GraficaCadena(0)

        # plt.plot()
        # for i in range(L):
            
        #     plt.scatter(Cadena[-1][4][i][0][0],Cadena[-1][4][i][0][1])
        # plt.show()



# # len(Cadena)
# # # Load
# # with open("/Users/isaias/Desktop/data.pkl", 'rb') as f:
# #     Cadena = pickle.load(f)





# Burnin=0#1000
# lagaux=1000
# logPost=np.ones(len(Cadena[Burnin::lagaux]))
# for i in range(len(Cadena[Burnin::lagaux])):
#     logPost[i]=Post(Cadena[Burnin::lagaux][i])
#     if i%100==0:
#         print(i/len(Cadena))
#         plt.plot(logPost[:i])
#         plt.show()
# plt.plot(logPost[:i])
# plt.show()


    
# for i in range(L):
#     CadenaMarginalBeta(i)




# GraficaCadena(0)
# GraficaCadena(1)
# GraficaCadena(2)
# GraficaCadena(31)
# GraficaCadena(32)
# GraficaCadena(33)





# # #######################################################################



# # Lindex=0


# IAT=np.zeros(L)
# GammaSim=np.zeros((len(Cadena),L))
# for i in range(len(Cadena)):
#     GammaSim[i,:]=Cadena[i][1][0]
# for l in range(L):
#     try:
#         IAT[l]=integrated_time(GammaSim[:,l])
#     except:
#         print()

####IATElegido=1000
# IATElegido=int(np.max(IAT))
# #### len(Cadena)/IATElegido
# for i in range(L):
#     plot_acf(GammaSim[:,i],lags=len(GammaSim)/5)
#     plt.show()

# # #######################################################################


# MuestraEfectiva=np.arange(len(Cadena))[IATElegido::IATElegido]

# print(len(MuestraEfectiva))
# for i in range(L):
#     plot_acf(GammaSim[IATElegido::IATElegido,i],lags=10)
#     plt.title(str(i))
#     plt.show()

        



# # #######################################################################


# pTGrupo=0

# for pTGrupo in range(pT):
#     for i in MuestraEfectiva:
#         plt.scatter(np.arange(L),Cadena[i][1][pTGrupo], alpha=0.2)
#         plt.axhline(1/L)
#         # plt.plot(np.arange(L),Cadena[i][1][pTGrupo], alpha=0.1)
#     plt.title(str(pTGrupo))
#     plt.show()

# # #######################################################################




#####Estimacion

# print(len(MuestraEfectiva))

# for j in range(pT):
#     PDF=DensPost(j)
    
#     x = np.linspace(-5,10,50)
#     y = np.linspace(-5,10,50)
#     Xg,Yg = np.meshgrid(x,y)
    
#     pdf=np.zeros(PDF.shape[1:])
#     var=np.zeros(PDF.shape[1:])
#     for i in range(Xg.shape[0]):
#         for j in range(Xg.shape[1]):
#             pdf[i,j]=np.mean(PDF[:,i,j])
#             var[i,j]=np.var(PDF[:,i,j])
    
    
    
#     plt.plot()
#     plt.contourf(Xg, Yg, pdf, cmap='viridis',levels=1000)
#     plt.colorbar()
#     plt.show()
    
#     plt.plot()
#     plt.contourf(Xg, Yg, var, cmap='viridis',levels=1000)
#     plt.colorbar()
#     plt.show()
    
#     plt.plot()
#     plt.contourf(Xg, Yg, np.sqrt(var)/pdf, cmap='viridis',levels=1000)
#     plt.colorbar()
#     plt.show()
    
    
#     CV=np.sqrt(var)/pdf
#     CV=1-(CV-np.min(CV))/CV
#     plt.imshow(pdf,alpha=CV,origin="lower",interpolation="bilinear")
#     plt.show()
    
    

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################



import numpy as np
import matplotlib.pyplot as plt

TimeCuts = 8
columnas=TimeCuts//2
Tmax = 10  # Define Tmax appropriately

fig, axes = plt.subplots(2, columnas, figsize=(12, 6), constrained_layout=True)

# Create a list of times
times = np.arange(0, Tmax + Tmax / TimeCuts, Tmax / TimeCuts)[:-1]

# Store one of the contour objects for the colorbar later
contour_ref = None

for idx, t in enumerate(times):

    row, col = divmod(idx, columnas)
    ax = axes[row, col]

    x = np.linspace(-5, 10, 50)
    y = np.linspace(-5, 10, 50)
    Xg, Yg = np.meshgrid(x, y)
    pdf0 = np.zeros(Xg.shape)
    
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            Obs = np.array((Xg[i, j], Yg[i, j], t))
            pdf0[i, j] = intensity(Obs)

    contour = ax.contourf(Xg, Yg, pdf0, cmap='viridis',
                          levels=np.linspace(0, np.ceil(np.max(PDF)), 10000), vmax=10)
    
    # Save reference to contour for colorbar
    if contour_ref is None:
        contour_ref = contour

    ax.set_title(f"t = {np.round(t, 2)}")

    # Only first column gets ylabel
    if col == 0:
        ax.set_ylabel("Y")
    else:
        ax.set_yticklabels([])

    # Only second row gets xlabel
    if row == 1:
        ax.set_xlabel("X")
    else:
        ax.set_xticklabels([])


# Add a colorbar between subplots (1,3) and (2,3)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Insert colorbar between the two rightmost axes
cax = inset_axes(axes[0, 2],  # anchor to top-right subplot
                 width="5%", height="100%", 
                 bbox_to_anchor=(1, 0, 1, 1),  # x, y, width, height in axes coords
                 bbox_transform=fig.transFigure,
                 loc='lower left',
                 borderpad=0)

fig.colorbar(contour_ref, cax=cax, orientation='vertical')

plt.show()

#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#####################################################



















