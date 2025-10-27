
############


import shapefile as shp #pip install pyshp
import os
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import invwishart,multivariate_normal,gamma,dirichlet,poisson,truncnorm

import numpy as np
import mpmath as mp
import scipy as sp
import matplotlib.pyplot as plt
import copy
from emcee.autocorr import integrated_time
from statsmodels.graphics.tsaplots import plot_acf
import pickle
from multiprocessing import Pool, cpu_count, get_context, Process
from scipy.optimize import linear_sum_assignment
import geopandas as gpd
from scipy.interpolate import griddata
###########

np.random.seed(1)



def Variables(valor):
    
    v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12=valor
    global Tmax,pT,xlim,Xlim,ylim,Ylim,SimPoisson,L,alpha0,IW,Gaus,GammaPrior
    Tmax = v1
    pT=v2
    xlim,Xlim=v3,v4
    ylim,Ylim=v5,v6
    SimPoisson=v7
    L=v8
    alpha0=v9
    IW=v10
    Gaus=v11
    GammaPrior=v12

def Variables2(valor):
    global Cadena
    Cadena=valor




def Variables3(valor):
    
    v1,v2=valor
    global mindate,CutTime
    mindate= v1
    CutTime=v2





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
        prod=Alpha[i+1]*Beta[i]
        if all(prod>0):
            p4+=mp.fsum(prod*np.log(Beta[i+1])+BetaInv(prod))
        else :
            prod=Alpha[i+1]*mp.matrix(Beta[i])
            p4+=mp.fsum(prod*np.log(Beta[i+1])+BetaInv(prod))        
        
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
    




def CadenaMarginalBeta(i):
        # plt.plot(Cadena[IATElegido::IATElegido,i])
        graf=np.zeros(len(Cadena))
        for j in range(len(Cadena)):
            graf[j]=Cadena[j][1][0][i]     
        plt.plot(graf)
        plt.title(str(i))
        plt.show()       


def CadenaMarginalBetaLabel(i,maps,Cadena):
        # plt.plot(Cadena[IATElegido::IATElegido,i])
        graf=np.zeros(len(Cadena))
        for j in range(len(Cadena)):
            graf[j]=Cadena[j][1][0][maps[j][i]]     
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
    fig, axs = plt.subplots(int(pT/2), 2)
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
    for i in range(1):
        ax.plot(GammaSim[:,i],label=str(i))
    # ax.legend(loc=2)
    # ax.plot(GammaSim[:,i])
    ax.set_title(Lindex)
    


def Grafmu():
    fig, axs = plt.subplots(pT, ClustTiempo)
    axs = axs.flatten()
    for i in range(L):
        Graf1mu(axs[i],np.arange(L)[i])
    plt.show()

def GraficaCadena(Gra):
    
    cmap = plt.cm.get_cmap('viridis', L)
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
        fig, axs = plt.subplots(2, pT//2)
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
            x = np.linspace(xlim,Xlim,20)
            y = np.linspace(ylim,Ylim,20)
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
            plt.xlim(xlim,Xlim)    
            plt.ylim(ylim,Ylim)
            npartX=20
            npartY=20
            DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
            DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
            try :
                shape=shp.Reader("./Mexico.shp") #http://geoportal.conabio.gob.mx/metadatos/doc/html/dest_2010gw.html
                for shape in shape.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    plt.plot(x,y,color="black")    
                # plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"])
                plt.plot(Trench["x"],Trench["y"],color="hotpink")
            except:
                pass

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
                
                
                x = np.linspace(xlim,Xlim,50)
                y = np.linspace(ylim,Ylim,50)
                Xg,Yg = np.meshgrid(x,y)
                pdf = np.zeros(Xg.shape)
                for s in range(Xg.shape[0]):
                    for t in range(Xg.shape[1]):
                        Obs=np.array((Xg[s,t], Yg[s,t]))
                        pdf[s,t] = multivariate_normal.pdf(Obs,mean=Cadena[-1][4][i][0], cov=Cadena[-1][4][i][1])
                plt.contour(Xg, Yg, pdf,levels=5,alpha=alphaGraf[i], color=cmap(i))
                        
                
                
            # plt.legend(loc=1)
            plt.show()
            contador+=1

    if Gra==32:
        Grafmu()
        
        
        
        
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




def SALT(beta):
    coord=np.random.choice(len(beta))
    aislado=np.random.choice(len(beta))
    bipropuesto=np.copy(beta)
    bi=beta[coord]
    logitbi=mp.log(mp.fdiv(bi,mp.fsum((1,-bi))))+np.random.normal(size=1,scale=1*bi*(1-bi) )[0]
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




def PostParalel(m):
    if m%(len(Cadena)//1000)==0:
        print(m/len(Cadena))
    X=Cadena[m]
    [Alpha,Beta,gammaParam,Z,Phi]=X
    p1=(alpha0-1)*np.log(Alpha[0])-Alpha[0]
    p2=(Alpha[0]/L-1)*np.sum(np.log(Beta[0]))+mp.log(BetaInv( np.repeat(Alpha[0]/L, L)))    
    p3=mp.fsum((Alpha[:-1]-1)*np.log(Alpha[1:])-Alpha[1:]-GammaVectorized(Alpha[:-1]))
    p4=0
    for i in range(pT-1):
        prod=Alpha[i+1]*Beta[i]
        if all(prod>0):
            p4+=mp.fsum(prod*np.log(Beta[i+1])+BetaInv(prod))
        else :
            prod=Alpha[i+1]*mp.matrix(Beta[i])
            p4+=mp.fsum(prod*np.log(Beta[i+1])+BetaInv(prod))        
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



def MeanVar(PDF):
    x = np.linspace(xlim,Xlim,50)
    y = np.linspace(ylim,Ylim,50)
    Xg,Yg = np.meshgrid(x,y)
    
    pdf=np.zeros(PDF.shape[1:])
    var=np.zeros(PDF.shape[1:])
    for i in range(Xg.shape[0]):
        for r in range(Xg.shape[1]):
            pdf[i,r]=np.mean(PDF[:,i,r])
            var[i,r]=np.var(PDF[:,i,r])
    return pdf,var


def DensPost(SelectedInterval,MuestraEfectiva,Cadena):

    x = np.linspace(xlim,Xlim,50)
    y = np.linspace(ylim,Ylim,50)
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
                    PDF[w,i,j] = PDF[w,i,j]+Cadena[r][2][SelectedInterval]*Cadena[r][1][SelectedInterval][s]*multivariate_normal.pdf(Obs,mean=Cadena[r][4][s][0], cov=Cadena[r][4][s][1])
        w+=1
 
    return PDF

        

def GrafPost(Opcion,pdf,var,vmax,CutPoint,j):
    # plt.plot()
    # plt.xlim(xlim,Xlim)    
    # plt.ylim(ylim,Ylim)
    CV=np.sqrt(var)/pdf
    CV=1-(CV-np.min(CV))/CV
    
    npartX=50
    npartY=50
    DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
    DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
    
    fig,ax1, = plt.subplots(nrows=1)
    
    ax1.set_xlim(xlim,Xlim)    
    ax1.set_ylim(ylim,Ylim)
    x = np.linspace(xlim,Xlim,50)
    y = np.linspace(ylim,Ylim,50)
    Xg,Yg = np.meshgrid(x,y)
    
    try:
        shape=shp.Reader("./Mexico.shp") #http://geoportal.conabio.gob.mx/metadatos/doc/html/dest_2010gw.html
        for shape in shape.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            ax1.plot(x,y,color="black")    
    except: 
        pass
    # plt.scatter(EarthquakesAux["Longitude"],EarthquakesAux["Latitude"])
    if Opcion==0:
        im1=ax1.contourf(Xg, Yg, pdf, cmap='viridis',levels=np.linspace(0,vmax,100 ))
        plt.colorbar(im1, ax=ax1)
    elif Opcion==1:
        im1=ax1.contourf(Xg, Yg, var, cmap='viridis',levels=1000)
        plt.colorbar(im1, ax=ax1)
    elif Opcion==2:
        im1=ax1.contourf(Xg, Yg, np.sqrt(var)/pdf, cmap='viridis',levels=1000)
        plt.colorbar(im1, ax=ax1)
    elif Opcion==3:
        im1=ax1.imshow(pdf,alpha=CV,origin="lower",interpolation="bilinear",extent=[DomX[0],DomX[-1],DomY[0],DomY[-1]],aspect=(DomX[1]-DomX[0])/(DomY[1]-DomY[0]),vmax=vmax)
        plt.colorbar(im1, ax=ax1)
    # plt.colorbar()
    try :
        ax1.plot(Trench["x"],Trench["y"],color="hotpink")
    except :
        pass
        
    ax1.set_title("Interval "+str(CutPoint[j])+"-"+str(CutPoint[j+1]))
    # ax1.set_title("Interval:"+str((mindate+timedelta(days=CutTime[j])))+"-"+str((mindate+timedelta(days=CutTime[j+1]))))
    # fig.show()

    plt.show()



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



# def PostMalla(PDF,opcion):
#     CutPoint=np.append((np.arange(0,pT)/pT)*Tmax,Tmax)
#     TimeCuts = pT
#     columnas=TimeCuts//2
    
#     fig, axes = plt.subplots(2, columnas, figsize=(12, 6), constrained_layout=True)
    
#     # Create a list of times
#     CutPoint
    
#     # Store one of the contour objects for the colorbar later
#     contour_ref = None
    
#     for idx, t in enumerate(CutPoint[:-1]):
#         row, col = divmod(idx, columnas)
#         ax = axes[row, col]
    
#         pdf,var=MeanVar(PDF[idx])
        
#         CV=np.sqrt(var)/pdf
#         CV=1-(CV-np.min(CV))/CV
#         npartX=50
#         npartY=50
#         DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
#         DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
        
#         ax.set_xlim(xlim,Xlim)    
#         ax.set_ylim(ylim,Ylim)
#         x = np.linspace(xlim,Xlim,50)
#         y = np.linspace(ylim,Ylim,50)
#         Xg,Yg = np.meshgrid(x,y) 
#         if opcion==0:
            
           
#             contour = ax.contourf(Xg, Yg, pdf, cmap='viridis',levels=np.linspace(0,np.ceil(np.max(PDF)),100 ))
#             if contour_ref is None:
#                 contour_ref = contour
#                 color_ticks = np.arange(0, np.ceil(np.max(PDF)) + 1, 1)
            
#         if opcion==1:
#             im1=ax.imshow(pdf,alpha=CV,origin="lower",interpolation="bilinear",extent=[DomX[0],DomX[-1],DomY[0],DomY[-1]],aspect=(DomX[1]-DomX[0])/(DomY[1]-DomY[0]),vmax=np.ceil(np.max(PDF)))
#             if contour_ref is None:
#                 contour_ref = im1
#                 color_ticks = np.arange(0, np.ceil(np.max(PDF)) + 1, 1)
                
#         if opcion==2:
#             pdf0 = np.zeros(Xg.shape)
#             for i in range(Xg.shape[0]):
#                 for j in range(Xg.shape[1]):
#                     Obs = np.array((Xg[i, j], Yg[i, j], t))
#                     pdf0[i, j] = intensity(Obs)

#             contour = ax.contourf(Xg, Yg, pdf0, cmap='viridis',levels=np.linspace(0, np.ceil(np.max(PDF)), 10000), vmax=10)
#             if contour_ref is None:
#                 contour_ref = contour
#                 color_ticks = np.arange(0, np.ceil(np.max(PDF)) + 1, 1)

    
#         # Save reference to contour for colorbar

    
#         ax.set_title(r"t $\in$("+str(CutPoint[idx])+","+str(CutPoint[idx+1])+")" )
    
#         # Only first column gets ylabel
#         if col == 0:
#             ax.set_ylabel("Y")
#         else:
#             ax.set_yticklabels([])
    
#         # Only second row gets xlabel
#         if row == 1:
#             ax.set_xlabel("X")
#         else:
#             ax.set_xticklabels([])
    
#     # Add a colorbar between subplots (1,3) and (2,3)
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
#     # Insert colorbar between the two rightmost axes
#     cax = inset_axes(axes[0, 1],  # anchor to top-right subplot
#                      width="5%", height="100%", 
#                      bbox_to_anchor=(1, 0, 1, 1),  # x, y, width, height in axes coords
#                      bbox_transform=fig.transFigure,
#                      loc='lower left',
#                      borderpad=0)
    
#     fig.colorbar(contour_ref, cax=cax, orientation='vertical', ticks=color_ticks)
    
#     plt.show()



def PostMalla(PDF,CV,opcion,mapa=0):
    CutPoint=np.append((np.arange(0,pT)/pT)*Tmax,Tmax)
    TimeCuts = pT
    columnas=TimeCuts//2

    
    fig, axes = plt.subplots(2, columnas, figsize=(12, 6), constrained_layout=True)
    # fig, axes = plt.subplots(2, columnas, figsize=(10, 4))
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.1, hspace=0.15)
    
    # Create a list of times
    CutPoint
    
    # Store one of the contour objects for the colorbar later
    contour_ref = None
    
    for idx, t in enumerate(CutPoint[:-1]):
        
       
        row, col = divmod(idx, columnas)
        ax = axes[row, col]
    
        pdf,var=MeanVar(PDF[idx])
        
        CV0=CV[idx]
        CV0=1-(CV0-np.min(CV))/CV0
        npartX=50
        npartY=50
        DomX=xlim+np.arange(npartX)/(npartX-1)*(Xlim-xlim)
        DomY=ylim+np.arange(npartY)/(npartY-1)*(Ylim-ylim)
        
        ax.set_xlim(xlim,Xlim)    
        ax.set_ylim(ylim,Ylim)
        x = np.linspace(xlim,Xlim,50)
        y = np.linspace(ylim,Ylim,50)
        Xg,Yg = np.meshgrid(x,y) 
        
        if mapa==0:
            vmax=np.ceil(np.max(PDF))
        elif mapa==1 :
            vmax=np.max(PDF)
            import geopandas as gpd
            shape=gpd.read_file("./dest23gw/dest23gw.shp")
            shape.plot(ax=ax, edgecolor='black', facecolor='none', zorder=10)            
            if idx==0:
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
                contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1), cmap='hot',alpha=0.5)
                # ax.clabel(contours, inline=True, fontsize=8)
                # plt.show()


            if idx==3:
                gdf = gpd.read_file("FFM2012.geojson")  # Replace with your file path
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
                contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1), cmap='hot',alpha=0.5)
                # ax.clabel(contours, inline=True, fontsize=8)
                # plt.show()


            if idx==3:
                gdf = gpd.read_file("FFM2014.geojson")  # Replace with your file path
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
                contours = ax.contour(xi, yi, zi, levels=np.arange(1,6.5,1), cmap='hot',alpha=0.5)
                # ax.clabel(contours, inline=True, fontsize=8)
                # plt.show()
            
            if idx==0:
                Mapa = './2001.geojson'
            if idx==1:
                Mapa = './2006.geojson'
            if idx==2:
                Mapa = './2009.geojson'
            if idx==3:
                Mapa = './2014.geojson'


            df = gpd.read_file(Mapa)
            p4=gpd.GeoSeries(df.geometry)
            p4.plot(ax=ax,facecolor="none", edgecolor='white', zorder=11)





    

        
        if opcion==0:
            
           
            contour = ax.contourf(Xg, Yg, pdf, cmap='viridis',levels=np.linspace(0,vmax,100 ))
            if contour_ref is None:
                contour_ref = contour
                if mapa==0:
                    color_ticks = np.arange(0, vmax + 1, 1)
                elif mapa==1:
                    color_ticks = color_ticks = np.linspace(0, vmax, 5)

            
        if opcion==1:
            # im1=ax.imshow(pdf,alpha=CV,origin="lower",interpolation="bilinear",extent=[DomX[0],DomX[-1],DomY[0],DomY[-1]],aspect=(DomX[1]-DomX[0])/(DomY[1]-DomY[0]),vmax=vmax)
            if mapa==0:
                im1=ax.imshow(pdf,alpha=CV0,origin="lower",interpolation="bilinear",extent=[DomX[0],DomX[-1],DomY[0],DomY[-1]],vmax=vmax)
            if mapa==1:
                im1=ax.imshow(pdf,alpha=CV0,origin="lower",interpolation="bilinear",extent=[DomX[0],DomX[-1],DomY[0],DomY[-1]],vmax=vmax,aspect=1.1)
            # ax.set_aspect('equal')
            # aspect_ratio = (DomY[-1] - DomY[0]) / (DomX[-1] - DomX[0])
            # ax.set_aspect(1 / aspect_ratio) 
            if contour_ref is None:
                contour_ref = im1
                if mapa==0:
                    color_ticks = np.arange(0, vmax + 1, 1)
                elif mapa==1:
                    color_ticks = color_ticks = np.linspace(0, vmax, 5)

        if opcion==2:
            pdf0 = np.zeros(Xg.shape)
            for i in range(Xg.shape[0]):
                for j in range(Xg.shape[1]):
                    Obs = np.array((Xg[i, j], Yg[i, j], t))
                    pdf0[i, j] = intensity(Obs)

            contour = ax.contourf(Xg, Yg, pdf0, cmap='viridis',levels=np.linspace(0, vmax, 10000), vmax=10)
            if contour_ref is None:
                contour_ref = contour
                if mapa==0:
                    color_ticks = np.arange(0, vmax + 1, 1)
                elif mapa==1:
                    color_ticks = color_ticks = np.linspace(0, vmax, 5)

    
        # Save reference to contour for colorbar

        if mapa==0:
            if opcion==2:
                ax.set_title(r"t $=$"+str(CutPoint[idx]) )
            else :
                ax.set_title(r"t $\in$("+str(CutPoint[idx])+","+str(CutPoint[idx+1])+")" )
        if mapa==1:
            ax.set_title(""+str((mindate+timedelta(days=CutTime[idx])))+"-"+str((mindate+timedelta(days=CutTime[idx+1]))))
    
        # Only first column gets ylabel
        if col == 0:
            if mapa==0:
                ax.set_ylabel("Y")
                
            if mapa==1:
                ax.set_ylabel("Latitude")

        else:
            ax.set_yticklabels([])
    
        # Only second row gets xlabel
        if row == 1:
            if mapa==0:
                ax.set_xlabel("X")
                
            if mapa==1:
                ax.set_xlabel("Longitude")
        else:
            ax.set_xticklabels([])
        plt.tight_layout()
    
    # Add a colorbar between subplots (1,3) and (2,3)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Insert colorbar between the two rightmost axes
    cax = inset_axes(axes[0, 1],  # anchor to top-right subplot
                     width="5%", height="100%", 
                     bbox_to_anchor=(1, 0, 1, 1),  # x, y, width, height in axes coords
                     bbox_transform=fig.transFigure,
                     loc='lower left',
                     borderpad=0)
    
    fig.colorbar(contour_ref, cax=cax, orientation='vertical', ticks=color_ticks)
    
    plt.show()










