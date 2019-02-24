#!/usr/bin/env python

import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
import math
I1=1  # values for torque 1,2,3 and I 1,2,3
I2=1
I3=1
T1=1
T2=0
T3=0
def RKB(Xo,Y1o,Y2o,Y3o,THEo,PHIo,LAMo,h,Xmax):
    x=Xo # time intial
    y1=Y1o # omega 1 intial
    y2=Y2o #omega 2 intial
    y3=Y3o # omega 3 intial
    the=THEo # theta intial value
    phi=PHIo # phi intial value
    lam=LAMo # lambda intial value
    m=(lambda y1,y2,y3,x: 1/I1*(I2-I3)*y2*y3+T1)  # function for dw1/dt
    mm=(lambda y2,y1,y3,x:1/I2*(I3-I1)*y3*y1+T2) #function for dw2/dt
    mmm=(lambda y3,y2,y1,x:1/I3*(I1-I2)*y1*y2+T3) #function for dw3/dt
    n=(lambda the,phi,lam,y1,y2,y3,x:y1*math.sin(lam)+y2*math.cos(lam))#function for d(theta)/dt
    nn=(lambda phi,the,lam,y1,y2,y3,x:(y2*math.sin(lam)-y1*math.cos(lam))/(math.sin(the)))#function for d(phi)/dt
    nnn=(lambda lam,phi,the,y1,y2,y3,x:y3+math.cos(the)/math.sin(the)*(y2*math.sin(lam)-y1*math.cos(lam)))# function for d(lam)/dt
    G=math.ceil(1/h*(Xmax-Xo))#I wanted my for loop to have to be an interger and to move up by an increment of h
    t=[]#time array
    s=[]#omega 1 array
    v=[]#omega 2 array
    z=[]#omega 3 array
    THE=[]#theta array
    PHI=[]#phi array
    LAM=[]#lambda array
    for i in range(1,G+2): #my for loops end was off by 2 so i just plus 2 to G
        #print((x,y1,y2,y3))#i just wanted to see what was happening with these values
        #print((x,the,phi,lam))
        t.append(x)# saves X in the t array vice versa for the other values
        s.append(y1)
        v.append(y2)
        z.append(y3)
        THE.append(the)
        PHI.append(phi)
        LAM.append(lam)
        k1the=n(the,phi,lam,y1,y2,y3,x)# k1 for theta... using runge kutta method
        k2the=n(the+.5*h*k1the,phi,lam, y1, y2,y3,x+.5*h)
        k3the=n(the+.5*h*k2the,phi,lam,y1,y2,y3,x+.5*h)
        k4the=n(the+h*k3the,phi, lam,y1, y2,y3,x+h)
        the=the+h/6*(k1the+2*k2the+2*k3the+k4the)#our next theta so theta(n+1)
        k1phi=nn(phi,the,lam,y1,y2,y3,x)#k1 for the phi
        k2phi=nn(phi+.5*h*k1phi,the,lam, y1, y2,y3,x+.5*h)
        k3phi=nn(phi+.5*h*k2phi,the,lam,y1,y2,y3,x+.5*h)
        k4phi=nn(phi+h*k3phi,the, lam,y1, y2,y3,x+h)
        phi=phi+h/6*(k1phi+2*k2phi+2*k3phi+k4phi)# this will be phi(n+1)
        k1lam=nnn(lam,phi,the,y1,y2,y3,x)
        k2lam=nnn(lam+.5*h*k1lam,phi,the, y1, y2,y3,x+.5*h)
        k3lam=nnn(lam+.5*h*k2lam,phi,the,y1,y2,y3,x+.5*h)
        k4lam=nnn(lam+h*k3lam,phi, the,y1, y2,y3,x+h)
        lam=lam+h/6*(k1lam+2*k2lam+2*k3lam+k4lam)# lam n+1
        k1y1=m(y1,y2,y3,x) #k1 for omega 1
        k2y1=m(y1+.5*h*k1y1,y2,y3,x+.5*h)
        k3y1=m(y1+.5*h*k2y1,y2,y3,x+.5*h)
        k4y1=m(y1+h*k3y1,y2,y3,x+h)
        k1y2=mm(y2,y1,y3,x)# k1 for omega 2
        k2y2=mm(y2+.5*h*k1y2,y1,y3,x+.5*h)
        k3y2=mm(y2+.5*h*k2y2,y1,y3,x+.5*h)
        k4y2=mm(y2+h*k3y2,y1,y3,x+h)
        k1y3=mmm(y3,y2,y1,x)#k3 for omega 3
        k2y3=mmm(y3+.5*h*k1y3,y2,y1,x+.5*h)
        k3y3=mmm(y3+.5*h*k2y3,y2,y1,x+.5*h)
        k4y3=mmm(y3+h*k3y3,y2,y1,x+h)
        y1=y1+h/6*(k1y1+2*k2y1+2*k3y1+k4y1) # omega 1 (n+1)
        y2=y2+h/6*(k1y2+2*k2y2+2*k3y2+k4y2)
        y3=y3+h/6*(k1y3+2*k2y3+2*k3y3+k4y3)
        x=Xo+i*h # time n+1

    fig, ax = plt.subplots()
    ax.plot(t, s)# plots omega 1 vs time
    ax.plot(t,v) # plots omega 2 vs time
    ax.plot(t,z) # plots omega 3 vs time
    #ax.plot(t, THE) # plots theta vs time
    #ax.plot(t,PHI) # plots phi vs time
    #ax.plot(t,LAM) # plots lambda vs time


    ax.set(xlabel='time', ylabel='omeagas',
           title='RKM "green=lambda" "orange=phi" "blue=theta"')
    ax.grid()

    fig.savefig("test.png")#this is for if I1=I2=I3 and T(y1,y2,y3,t)
   
RKB(0, 1, 0,0,(math.pi)/2,0,(math.pi)/2,.01,2)    #Xo, Y1o,Y2o,Y30,THEo,phio,lamo, h , Xmax
plt.show()
