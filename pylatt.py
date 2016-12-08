'''
A simple lattice code in Python
originally from Sam Krinsky's python code
re-write in OOP style
2012-11-17

2012-12-10: simple parse to read mad8 input
2012-12-20: clean namespace, remove "from numpy import ..."
2013-04-17: 6d transport matrix replace 3d matrix
2013-09-26: new kamp format to support 1st and 2nd kickmaps
            remove txt2py, becuase its function can be realized
            by txt2latt()+ savefile()
2013-10-04: switch off insertion device, the optics without ID can
            be set as the reference of beta-function correction
2013-10-05: measure beat-beat response matrix
'''

__version__ = 1.0
__author__ = 'Yongjun Li, yli@bnl.gov or mpyliyj@hotmail.com'

import time
import string
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
from matplotlib.cbook import flatten
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

#global parameter: speed of light
csp = 299792458.

class drif():
    '''
    class: drif - define drift space with given name and length
    usage: D1 = drif(name='D1',L=1.0)

    Function list:
    __init__()
    __repr__()
    put()
    update()
    transmatrix()
    twissmatrix()

    Parameter list:
    name:      element name
    L:         length
    tm:        transport matrix 6x6
    tx,ty:     twiss matrics 3x3 for x and y plane
    '''
    def __init__(self,name='D',L=0):
        self.name = name
        self.L = float(L)
        self.update()

    def __repr__(self):
        return '%s: %s, L = %g'%(self.name,self.__class__.__name__,self.L)

    def put(self, field, new_value):
        '''
        put a new value on an instance's attribute
        usage: instance.put(field_name, new_field_value)

        If use . operator to modify an instance's attribute, 
        matrices won't change untill call update().
        if use put(), its matrices will be updated 
        automatically (recommended).
        '''
        if hasattr(self,field):
            setattr(self,field,new_value)
            if field != 'name':
                self.update()
        else:
            raise RuntimeError("'%s' object has no attribute '%s'"
                               %(self.__class__.__name__,field))
    
    def update(self):
        '''
        update transport and Twiss matrices using current parameters
        '''
        self.transmatrix()
        self.twissmatrix()

    def transmatrix(self):
        '''
        calculate transport matrix
        '''
        self.tm = np.mat(np.eye(6))
        self.tm[0,1] = self.L
        self.tm[2,3] = self.L

    def twissmatrix(self):
        '''
        from transport matrix to twiss matrix
        '''
        self.tx = trans2twiss(self.tm[0:2,0:2])
        self.ty = trans2twiss(self.tm[2:4,2:4])


class moni(drif):
    '''
    monitor (BPM) with a default length 0
    usage: moni(name='bpm1',L=0)
    '''
    def __init__(self,name='M',L=0):
        self.L = float(L)
        self.name = str(name)
        self.update()


class kick(drif):
    '''
    kick with a default length 0
    usage: kick(name='kick',L=0)
    '''
    def __init__(self,name='C',L=0,hkick=0,vkick=0):
        self.L = float(L)
        self.name = str(name)
        self.hkick = hkick
        self.vkick = vkick
        self.update()
    def __repr__(self):
        return '%s: %s, L = %g, hkick = %g, vkick = %g'%(
            self.name,self.__class__.__name__,self.L,self.hkick,self.vkick)


class aper(drif):
    '''
    class: aper - define a physical aperture with two dimension constraints
    usage: aper(name='ap1',aper=[-0.02,0.02,-0.01,0.01])
    
    notice: aperture is specfied by 4 parameters in sequence:
    [x_min,x_max,y_min,y_max]
    correctness of aperture configuration can be self-checked
    by calling selfCheck()
    '''
    def __init__(self,name='A',L=0,aper=[-1.,1.,-1.,1.]):
        self.L = float(L)
        self.name = str(name)
        self.aper = np.array(aper)
        self.selfCheck()
        self.update()

    def selfCheck(self):
        '''
        check aperture dimension, if OK, return True, otherwise reture False
        aperture must have 4 parameters in a format: [x_min,x_max,y_min,y_max] 
        '''
        if len(self.aper) != 4:
            print('aperture dimensions must be 4')
            return False
        if self.aper[0]>=self.aper[1] or self.aper[2]>=self.aper[3]:
            print('no aperture to let particle go through')
            print('correct aperture format: [x_min,x_max,y_min,y_max]')
            return False
        return True


class sext(drif):
    '''
    class: sext - define setupole with length and K2
    usage: sext(name='S1',L=0.25,K2=1.2)
    
    K2 = 1/(B*Rho)*d(dB/dx)/dx - MAD definition
    '''
    def __init__(self,name='S',L=0,K2=0):
        self.L = float(L)
        self.name = name
        self.K2 = float(K2)
        self.update()
   
    def __repr__(self):
        return '%s: %s, L = %g, K2 = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.K2)


class kmap(drif):
    '''
    class: kmap - define a RADIA kickmap from an external text file
    usage: kmap(name='IVU29',L=1.2,kmap2fn='cwd/ivu29_kmap.txt',E=3)

    kmap1fn: 1st kickmap file name with directory info
    kmap2fn: 2nd kickmap file name with directory info
    E: beam energy in GeV

    1. kickmap file itself includes a length. This definition will scale
    with the integral to the actual device length, i.e. L.

    2. kickmap includes the magnetic field [Tm/T2m2] or kicks [micro-rad]
    normalized with a certain beam energy (3Gev by default). The kick unit
    is re-normalized to field unit.
    '''
    def __init__(self,name='INS', L=0,
                 kmap1fn=None, kmap2fn=None,
                 E=3, nk=100):
        self.name = name
        self.L = float(L)
        self.kmap1fn = kmap1fn
        self.kmap2fn = kmap2fn
        self.E = float(E)
        self.nk = int(nk)
        self.update()

    def __repr__(self):
        return "%s: %s, L = %g, E =%8.4f, kmap1fn = '%s', kmap2fn = '%s'"%(
            self.name,self.__class__.__name__,self.L,self.E,
            self.kmap1fn,self.kmap2fn)

    def transmatrix(self):
        '''
        calculate transport matrix from kick map files
        first, read kick maps from given kick map files
        then, if kick map unit is [micro-rad], 
        it will be un-normailized by beam energy, and reset its unit as [field]
        '''
        BRho = self.E*1e9/csp
        if self.kmap1fn:
            self.kmap1 = readkmap(self.kmap1fn, self.L)
            if self.kmap1['unit'] == 'kick':
                self.kmap1['kx'] = self.kmap1['kx']*1e-6*BRho
                self.kmap1['ky'] = self.kmap1['ky']*1e-6*BRho
                self.kmap1['unit'] = 'field'
        else:
            self.kmap1 = None
        if self.kmap2fn:
            self.kmap2 = readkmap(self.kmap2fn, self.L)
            if self.kmap2['unit'] == 'kick':
                self.kmap2['kx'] = self.kmap2['kx']*1e-6*BRho*BRho
                self.kmap2['ky'] = self.kmap2['ky']*1e-6*BRho*BRho
                self.kmap2['unit'] = 'field'
        else:
            self.kmap2 = None
        self.tm = kmap2matrix(self.kmap1, self.kmap2, self.E, self.nk)
        

class quad(drif):
    '''
    class: quad - define a quadrupole with given length and K1
    usage: quad(name='Q1',L=0.5,K1=0.23)
    '''
    def __init__(self,name='Q',L=0,K1=0):
        self.name = name
        self.L = float(L)
        self.K1 = float(K1)
        self.update()

    def __repr__(self):
        return '%s: %s, L = %g, K1 = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.K1)

    def transmatrix(self):
        self.tm = np.mat(np.eye(6))
        if self.K1 > 0:
            k = np.sqrt(self.K1)
            p = k*self.L
            self.tm[0:2,0:2] = [[np.cos(p),np.sin(p)/k],
                                [-k*np.sin(p),np.cos(p)]]
            self.tm[2:4,2:4] = [[np.cosh(p),np.sinh(p)/k],
                                [k*np.sinh(p),np.cosh(p)]]
        elif self.K1 < 0:
            k = np.sqrt(-self.K1)
            p = k*self.L
            self.tm[0:2,0:2] = [[np.cosh(p),np.sinh(p)/k],
                                [k*np.sinh(p),np.cosh(p)]]
            self.tm[2:4,2:4] = [[np.cos(p),np.sin(p)/k],
                                [-k*np.sin(p),np.cos(p)]]
        else:
            self.tm[0,1] = self.L
            self.tm[2,3] = self.L


class bend(drif):
    '''
    class: bend - define a bend (dipole) with given parmeters
    usage: bend(name='B1',L=1,angle=0.2,e1=0.1,e2=0.1,K1=0,K2=0)
    
    notice: it can have quadrupole and sextupole gradients integrated
    '''
    def __init__(self,name='B',L=1,angle=1e-9,e1=0,e2=0,K1=0,K2=0):
        self.name = name
        self.L = float(L)
        self.angle = float(angle)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.K1 = float(K1)
        self.K2 = float(K2)
        self.R = self.L/self.angle
        self.update()

    def __repr__(self):
        return '%s: %s, L = %g, angle = %15.8f, e1 = %15.8f, e2 = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.angle,self.e1,self.e2)

    def transmatrix(self):
        kx = self.K1+1/(self.R*self.R)
        self.tm = np.mat(np.eye(6))
        if kx > 0:
            k = np.sqrt(kx)
            p = k*self.L
            self.tm[0:2,0:2] = [[np.cos(p),np.sin(p)/k],
                                [-k*np.sin(p),np.cos(p)]]
            self.tm[0:2,5:6] = [[(1-np.cos(p))/(self.R*kx)],
                                [np.sin(p)/(self.R*k)]]
            self.tm[4,0] = self.tm[1,5]
            self.tm[4,1] = self.tm[0,5]
            self.tm[4,5] = (k*self.L-np.sin(p))/self.R/self.R/k/kx
        elif kx < 0:
            k = np.sqrt(-kx)
            p = k*self.L
            self.tm[0:2,0:2] = [[np.cosh(p),np.sinh(p)/k],
                                [k*np.sinh(p),np.cosh(p)]]
            self.tm[0:2,5:6] = [[(np.cosh(p)-1)/(-self.R*kx)],
                                [np.sinh(p)/(self.R*k)]]
            self.tm[4,0] = self.tm[1,5]
            self.tm[4,1] = self.tm[0,5]
            self.tm[4,5] = (k*self.L-np.sinh(p))/self.R/self.R/k/kx
        else:
            self.tm[0,1] = self.L

        if self.K1 > 0:
            k = np.sqrt(self.K1)
            p = k*self.L
            self.tm[2:4,2:4] = [[np.cosh(p),np.sinh(p)/k],
                                [k*np.sinh(p),np.cosh(p)]]
        elif self.K1 < 0:
            k = np.sqrt(-self.K1)
            p = k*self.L
            self.tm[2:4,2:4] = [[np.cos(p),np.sin(p)/k],
                                [-k*np.sin(p),np.cos(p)]]
        else:
           self.tm[2,3] = self.L

        if self.e1 != 0.:
            m1 = np.mat(np.eye(6))
            m1[1,0] = np.tan(self.e1)/self.R
            m1[3,2] = -m1[1,0]
            self.tm = self.tm*m1

        if self.e2 != 0.:
            m2 = np.mat(np.eye(6))
            m2[1,0] = np.tan(self.e2)/self.R
            m2[3,2] = -m2[1,0]
            self.tm = m2*self.tm


class wigg(drif):
    '''
    class: wigg - define a simple model for wiggler with given length and field
    usage: wigg(name='WG',L=1,Bw=1.5,E=3)

    Bw: average wiggler field strength within one half period
    E: beam energy in GeV
    '''
    def __init__(self,name='W',L=0,Bw=0,E=3):
        self.name = name
        self.L = float(L)
        self.Bw = float(Bw)
        self.E = float(E)
        self.update()

    def transmatrix(self):
        rhoinv = self.Bw/(self.E*1e9)*csp
        k = abs(rhoinv)/np.sqrt(2)
        p = k*self.L
        self.tm = np.mat(np.eye(6))
        self.tm[0,1] = self.L
        self.tm[2:4,2:4] = [[np.cos(p),np.sin(p)/k],
                            [-k*np.sin(p),np.cos(p)]]

    def __repr__(self):
        return '%s: %s, L = %g, Bw = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.Bw)


class beamline():
    '''
    class: beamline - define a beamline with a squence of magnets
    usage: beamline(bl,twx0,twy0,dx0,N=1,E=3)
    
    twx0, twy0, dx: Twiss paramters at the starting point
                    3x1 matrix
    notice: 
    '''
    def __init__(self,bl,twx0=[10,0,0.1],twy0=[10,0,0.1],
                 dx0=[0,0,0],em=[1000.,1000.],sige=1e-2,N=1,E=3):
        self.E = float(E)
        self.N = int(N)
        self.bl = [ele for ele in flatten(bl)]*self.N
        try:
            self.twx = np.mat(twx0).reshape(3,1)
            self.twy = np.mat(twy0).reshape(3,1)
            self.dx = np.mat(dx0).reshape(3,1)
        except:
            raise RuntimeError('initial Twiss and dispersion dimension must be 3')
        self.twx[2,0] = (1+self.twx[1,0]*self.twx[1,0])/self.twx[0,0]
        self.twy[2,0] = (1+self.twy[1,0]*self.twy[1,0])/self.twy[0,0]
        self.dx[2,0] = 1
        self.emitx = em[0]
        self.emity = em[1]
        self.sige = sige
        self.update()

    def update(self):
        self.twx,self.twy = self.twx[:,0],self.twy[:,0]
        self.dx = self.dx[:,0]
        self.mux,self.muy = [0.],[0.]
        self.spos()
        self.twiss()
        self.disp()
        self.sigx = [np.sqrt(self.betax[m]*self.emitx*1e-9)
                     +abs(self.dx[0,m])*self.sige for m in range(len(self.s))]
        self.sigy = [np.sqrt(self.betay[m]*self.emity*1e-9)
                     for m in range(len(self.s))]

    def __repr__(self):
        s = ''
        s += '\n'+'-'*9*11+'\n'
        s += 9*'%11s'%('s','betax','alfax','mux','dx','dxp','betay','alfay','muy')+'\n'
        s += '-'*9*11+'\n'
        s += (9*'%11.3e'+'\n')%(self.s[0],self.twx[0,0],self.twx[1,0],self.mux[0], \
                                self.dx[0,0],self.dx[1,0],self.twy[0,0],\
                                self.twy[1,0],self.muy[0])
        s += (9*'%11.3e'+'\n')%(self.s[-1],self.twx[0,-1],self.twx[1,-1],self.mux[-1], \
                                self.dx[0,-1],self.dx[1,-1],self.twy[0,-1],\
                                self.twy[1,-1],self.muy[-1])+'\n'
        s += 'Tune: nux = %11.3e, nuy = %11.3e\n'%(self.nux,self.nuy)+'\n'
        return s

    def spos(self):
        '''
        get s coordinates for each element
        '''
        s = [0]
        for elem in self.bl:
            s.append(s[-1]+elem.L)
        self.s = np.array(s)

    def twiss(self):
        '''
        get twiss parameters along s. twx/y are matrix format,
        betax/y, alfax/y are 1D array, duplicate data are kept for convinence
        '''
        for elem in self.bl:
            mx = np.take(np.take(elem.tm,[0,1,5],axis=0),[0,1,5],axis=1)
            my = np.take(np.take(elem.tm,[2,3,5],axis=0),[2,3,5],axis=1)
            self.mux = np.append(self.mux,self.mux[-1]+phasetrans(mx,self.twx[:,-1]))
            self.muy = np.append(self.muy,self.muy[-1]+phasetrans(my,self.twy[:,-1]))
            self.twx = np.append(self.twx,twisstrans(elem.tx,self.twx[:,-1]),axis=1)
            self.twy = np.append(self.twy,twisstrans(elem.ty,self.twy[:,-1]),axis=1)
        self.betax = self.twx.getA()[0]
        self.betay = self.twy.getA()[0]
        self.alfax = self.twx.getA()[1]
        self.alfay = self.twy.getA()[1]
        self.gamax = self.twx.getA()[2]
        self.gamay = self.twy.getA()[2]
        self.nux = self.N*self.mux[-1]
        self.nuy = self.N*self.muy[-1]

    def getElements(self,types=['drif','bend','quad','sext','moni',\
                                 'aper','kick','wigg','kmap'],\
                    prefix='',suffix='',include ='',lors='list'):
        '''
        get all elements with same types and start with the specified string
        if lors == 'list' return a list, otherwise return a set
        used for optimization
        '''
        el = [a for a in self.bl 
              if a.__class__.__name__ in types  \
                  and a.name.startswith(prefix) \
                  and a.name.endswith(suffix)\
                  and include in a.name]
        if lors == 'list':
            return el
        else:
            return set(el) 
        
    def getSIndex(self,s0):
        '''
        get the nearest element index which is neighboring to s0
        '''
        s0 = float(s0)
        if s0 < 0:
            return 0
        elif s0 > self.s[-1]:
            return -1
        else:
            m = np.nonzero(np.array(self.s) > s0)[0][0]
            if s0-self.s[m-1] > self.s[m]-s0:
                return m
            else:
                return m-1

    def getIndex(self,types,prefix='',suffix='',exitport=True):
        '''
        get all element index with the given types and 
        starts/ends with the specified string
        types:    str/list-like, element types
        prefix:   str, which is the string element name starts with
        suffix:   str, which is the string element name ends with
        exitport: bool, entrance or exit, if True exit, otherwise entrance
        '''
        a = []
        for i,el in enumerate(self.bl):
            if el.__class__.__name__ in types  \
                    and el.name.startswith(prefix) \
                    and el.name.endswith(suffix):
                if exitport:
                    a.append(i+1)
                else:
                    a.append(i)
        return a

    def disp(self):
        '''
        get dispersion along s
        '''
        for elem in self.bl:
            mx = np.take(np.take(elem.tm,[0,1,5],axis=0),[0,1,5],axis=1)
            self.dx = np.append(self.dx,mx*self.dx[:,-1],axis=1)
        self.etax = self.dx.getA()[0]
        self.etaxp = self.dx.getA()[1]

    def pltmag(self,unit=1.0,alfa=1):
        '''
        plot magnets along s
        '''
        sh = 0.75*unit #sext
        qh = 1.0*unit  #quad
        bh = 0.5*unit  #bend
        kh = 0.25*unit #kmap - insertion device
        mh = 0.20*unit #moni - bpm
        s = [0.]
        ax = plt.gca()
        for e in self.bl:
            if e.__class__.__name__ == 'drif':
                pass
            elif e.__class__.__name__ == 'quad':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],-qh),e.L,2*qh,facecolor='y',alpha=alfa))
            elif e.__class__.__name__ == 'sext':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],-sh),e.L,2*sh,facecolor='r',alpha=alfa))
            elif e.__class__.__name__ == 'bend':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],-bh),e.L,2*bh,facecolor='b',alpha=alfa))
            elif e.__class__.__name__ == 'kmap':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],-kh),e.L,2*kh,facecolor='g',alpha=alfa))
            elif e.__class__.__name__ == 'moni':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],-mh),e.L,2*mh,facecolor='m',alpha=alfa))
            else:
                pass
            s += [s[-1]+e.L]
        plt.plot(s, np.zeros(len(s)),'k',linewidth=1.5)

    def plttwiss(self,srange=None,save=False,fn='twiss.png'):
        '''
        plot twiss and dispersion
        parameters: save:      bool, if true, save the plot into the file
                               with the given name as 'fn = xxx.xxx'
                    fn:        str,  specify the file name if save is true 
                    srange:    float/list, a list with two elements to define
                               [s-begin, s-end], whole beam-line by default
        updated on 2013-12-11, add srange option suggested by S. Krinsky
        '''
        plt.figure()
        plt.plot(self.s,self.betax,'r',linewidth=2,label=r'$\beta_x$')
        plt.plot(self.s,self.betay,'b',linewidth=2,label=r'$\beta_y$')
        plt.plot(self.s,self.etax*10,'g',linewidth=2,label=r'$10\times\eta_x$')
        if srange:
            size = plt.axis()
            plt.axis([srange[0],srange[1],size[2],size[3]])
        self.pltmag(unit=max(self.betax)/20)
        plt.xlabel(r'$s$ (m)')
        plt.ylabel(r'$\beta$ and $\eta$ (m)')
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=3, mode="expand",borderaxespad=0.)
        #plt.title('Twiss Parameters')
        if save:
            plt.savefig(fn)
        plt.show()

    def pltsigma(self,srange=None,save=False,fn='sigma.png'):
        '''
        plot transverse beam size along s
        parameters: save:      bool, if true, save the plot into the file
                               with the given name as 'fn = xxx.xxx'
                    fn:        str,  specify the file name if save is true 
                    srange:    float/list, a list with two elements to define
                               [s-begin, s-end], whole beam-line by default
        '''
        plt.figure()
        unit = 1000.
        plt.plot(self.s,np.array(self.sigx)*unit,
                 linewidth=2,label=r'$\sigma_x$')
        plt.plot(self.s,np.array(self.sigy)*unit,
                 linewidth=2,label=r'$\sigma_y$')
        if srange:
            size = plt.axis()
            plt.axis([srange[0],srange[1],size[2],size[3]])
        self.pltmag(unit=max(self.sigx)*unit/20)
        plt.xlabel(r'$s$ (m)')
        plt.ylabel(r'$\sigma_x$ (mm)')
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=2, mode="expand",borderaxespad=0.)
        if save:
            plt.savefig(fn)
        plt.show()


    def savefile(self,fn='savefile.py'):
        '''
        save current beamline to a python file
        '''
        ele = list(set(self.bl))
        ele = sorted(ele, key=lambda el: el.__class__.__name__+el.name)
        fid = open(fn,'w')
        fid.write('import pylatt as latt\n\n')
        fid.write('# === Element definition:\n')
        for e in ele:
            if e.__class__.__name__ == 'drif':
                s = '{0} = latt.drif("{1}",L={2})\n'.\
                    format(e.name,e.name,e.L)
            elif e.__class__.__name__ == 'quad':
                s = '{0} = latt.quad("{1}",L={2},K1={3})\n'.\
                    format(e.name,e.name,e.L,e.K1)
            elif e.__class__.__name__ == 'sext':
                s = '{0} = latt.sext("{1}",L={2},K2={3})\n'.\
                    format(e.name,e.name,e.L,e.K2)
            elif e.__class__.__name__ == 'bend':
                s = '{0} = latt.bend("{1}",L={2},angle={3},e1={4},e2={5},K1={6},K2={7})\n' \
                    .format(e.name,e.name,e.L,e.angle,e.e1,e.e2,e.K1,e.K2)
            elif e.__class__.__name__ == 'kmap':
                s = '{0} = latt.kmap("{1}",L={2},kmap1fn="{3}",kmap2fn="{4}",E={5})\n' \
                    .format(e.name,e.name,e.L,e.kmap1fn,e.kmap2fn,e.E) \
                    .replace('"None"','None')
            elif e.__class__.__name__ == 'moni':
                s = '{0} = latt.moni("{1}",L={2})\n'.format(e.name,e.name,e.L)
            elif e.__class__.__name__ == 'aper':
                s = '{0} = latt.aper("{1}",L={2},aper=[{3},{4},{5},{6}])\n' \
                    .format(e.name,e.name,e.L,e.aper[0],e.aper[1],e.aper[2],e.aper[3])
            elif e.__class__.__name__ == 'kick':
                s = '{0} = latt.kick("{1}",L={2},hkick={3},vkick={4})\n'\
                    .format(e.name,e.name,e.L,e.hkick,e.vkick)
            else:
                raise RuntimeError('unknown element type: {0}'.\
                                   format(e.__class__.__name__))
            fid.write(s)
        fid.write('\n# === Beam Line sequence:\n')
        BL = 'BL = ['
        for e in self.bl:
            BL += e.name + ', '
            if len(BL) > 72:
                fid.write(BL+'\n')
                BL = '   '
        BL = BL[0:-2] + ']\n'
        fid.write(BL)
        fid.write('ring = latt.cell(BL)\n')
        fid.close()

    def saveLte(self,fn='temp.lte',simplify=False):
        '''
        save beamline into ELEGANT format file
        fn:       file's name for saving lattice
        simplify: bool, if True, ignore all drifts with zero length
        '''
        fid = open(fn,'w')
        fid.write('! === Convert from Yongjun Li\'s pylatt input \n')
        fid.write('! === Caution: kickmaps need to be adjusted manually\n\n')
        fid.write('! === Element definition:\n')
        eset = list(set(self.bl))
        eset = sorted(eset, key=lambda ele: ele.__class__.__name__+ele.name)
        for ei in eset:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                aele = ei.__repr__() \
                    .replace('sext','ksext') \
                    .replace('quad','kquad') \
                    .replace('bend','csbend')+'\n'
                fid.write(aele)
        fid.write('\n')
        linehead = '! === Beam line sequence:\nring: LINE = ('
        for ei in self.bl:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                linehead += ei.name+', '
            if len(linehead) > 72:
                fid.write(linehead+'&\n')
                linehead = '  '
        fid.write(linehead[:-2]+')\n\n')
        fid.write('use, ring\nreturn\n')
        fid.close()

    def saveTesla(self,fn='temp.lte',simplify=False):
        '''
        save beamline into Lingyun Yang's TESLA format
        fn:       file's name for saving lattice
        if simplify is true, ignore all drifts with zero length
        '''
        fid = open(fn,'w')
        fid.write('# === Convert from Yongjun Li\'s pylatt input \n\n')
        fid.write('# === Element definition:\n')
        eset = list(set(self.bl))
        eset = sorted(eset, key=lambda ele: ele.__class__.__name__+ele.name)
        for ei in eset:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                aele = ei.__repr__() \
                    .replace('sext','Sextupole') \
                    .replace('drif','Drift') \
                    .replace('quad','Quadrupole') \
                    .replace('bend','Dipole') \
                    .replace('moni','BPM') \
                    .replace('E =','KM_PREF =') \
                    .replace('kmap1fn','KM_FILE_EXTRA1')\
                    .replace('kmap2fn','KM_FILE')\
                    .replace('kmap,','ID,') +';\n'
                fid.write(aele)
        fid.write('\n')
        linehead = '# === Beam line sequence:\nring: LINE = ('
        for ei in self.bl:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                linehead += ei.name+', '
            if len(linehead) > 72:
                fid.write(linehead+'\n')
                linehead = '  '
        fid.write(linehead[:-2]+');\n\n')
        fid.write('setup, line="ring", shared=false;\n')
        fid.close()

    def getMatLine(self):
        '''
        prepare matrix and thin-lens kick for tracking
        mlist is a list of transport matrices between two
        adjacent npnlinear elements (sext and kmap)
        klist is a list of nonlinear element position index
        If dipole has sext-like component, don't use this function
        '''
        mlist = []
        nk = [0]
        for i,ele in enumerate(self.bl):
            if ele.__class__.__name__ in ['sext','kmap']:
                nk.append(i)
        nk.append(len(self.bl))
        for i,j in enumerate(nk[:-1]):
            R = np.mat(np.eye(6))
            for e in self.bl[nk[i]:nk[i+1]]:
                if e.__class__.__name__ not in ['sext','kmap']:
                    R = e.tm*R
            mlist.append(R)
        klist = nk[1:-1]
        self.mlist = mlist
        self.klist = klist

    def eletrack(self, x0, end_index=None):
        '''
        element by element track
        '''
        x0 = np.reshape(x0,(6,-1))
        if end_index and end_index<=len(self.bl):
            se = end_index
        else:
            se = len(self.bl)
        x = np.zeros((se+1,x0.shape[0],x0.shape[1]))
        x[0] = x0 
        for i in range(se):
            x0[:6] = elempass(self.bl[i],x0[:6])
            x[i+1] = x0
        return x

    def matrack(self, x0, nturn=1):
        '''
        implement a simple matrix tracking
        x0 is 6xm data representing particle coordinator
        '''
        x0 = np.reshape(x0,(6,-1))
        if not hasattr(self,'mlist'):
            self.getMatLine()
        for n in range(nturn):
            for i,k in enumerate(self.klist):
                x0[:6] = self.mlist[i]*x0[:6]
                x0 = chkap(x0)
                x0[:6] = elempass(self.bl[k],x0[:6])
                x0 = chkap(x0)
            x0[:6] = self.mlist[-1]*x0[:6]
            x0 = chkap(x0)
        return x0

    def cmbdrf(self):
        '''
        combine neighboring drifts to form a simplified beamline
        '''
        drfidx = 0
        newline = []
        for ele in self.bl:
            if ele.__class__.__name__ != 'drif':
                newline.append(ele)
            else:
                if not newline or newline[-1].__class__.__name__ != 'drif':
                    drfidx += 1
                    dftname = 'D'+string.zfill(drfidx,4)
                    d = drif(dftname,L=ele.L)
                    newline.append(d)
                else:
                    newline[-1].put('L',newline[-1].L+ele.L)
        return newline
                    

class cell(beamline):
    '''
    Periodical solution for a beamline
    '''
    def __init__(self,bl,N=1,E=3.,kcouple=0.01):
        '''
        bl:  an element list to compose a beamline
             each elements must be a pre-defined instances of magnet
        N:   int, number of period
        E:   float, beam energy in GeV (3 by default)
        kcouple: linear transverse coupling coefficient.
        '''
        self.E = float(E)
        self.N = int(N)
        self.bl = [ele for ele in flatten(bl)]
        self.kcouple = kcouple
        self.update()

    def update(self,rad=False,chrom=False,ndt=False):
        '''
        refresh periodical cell parameters
        rad = True, calculate all radiation related parameters
        chrom = True, calculate chromaticity
        ndt = True, calculate nonlinear driving terms
        '''
        self.spos()
        self.isstable,self.R,self.twx,self.twy,self.dx = period(self.bl)
        if self.isstable:
            self.mux,self.muy = [0.],[0.]
            self.twiss()
            self.disp()
            if rad:
                self.rad()
            if chrom:
                self.chrom()
            if ndt: #nonlinear driving terms
                self.geth1()
                self.geth2()
        else:
            print('\n === Warning: no stable linear optics === \n')

    def __repr__(self):
        '''
        List some main parameters 
        '''
        s = ''
        s += '\n'+'-'*9*11+'\n'
        s += 9*'%11s'%('s','betax','alfax','mux','dx','dxp','betay','alfay','muy')+'\n'
        s += '-'*9*11+'\n'
        s += (9*'%11.3e'+'\n')%(self.s[0],self.twx[0,0],self.twx[1,0],self.mux[0], \
                                 self.dx[0,0],self.dx[1,0],self.twy[0,0],\
                                 self.twy[1,0],self.muy[0])
        s += (9*'%11.3e'+'\n')%(self.s[-1],self.twx[0,-1],self.twx[1,-1],self.mux[-1], \
                                 self.dx[0,-1],self.dx[1,-1],self.twy[0,-1],\
                                 self.twy[1,-1],self.muy[-1])+'\n'
        s += 'Tune: nux = %11.3e, nuy = %11.3e\n'%(self.nux,self.nuy)+'\n'
        if hasattr(self,'chx0'):
            s += 'uncorrected chromaticity: chx0 = %11.3e, chy0 = %11.3e\n'%(
                self.chx0,self.chy0)+'\n'
            s += 'corrected chromaticity:   chx  = %11.3e, chy  = %11.3e\n'%(
                self.chx,self.chy)+'\n'
        if hasattr(self,'emitx'):
            s += 'Horizontal emittance [nm.rad] = %11.3e\n'%self.emitx+'\n'
            s += 'Momentum compactor alphac = %11.3e\n'%self.alphac+'\n'
            s += 'Radiation damping factor: D = %11.3e\n'%self.D+'\n'
            s += 'Radiation damping partition factors: Jx = %11.3e, Jy = %11.3e, Je = %11.3e\n'%(self.Jx,self.Jy,self.Je)+'\n'
            s += 'Radiation damping time [ms]: taux = %11.3e, tauy = %11.3e, taue = %11.3e\n'%(self.taux,self.tauy,self.taue)+'\n'
            s += 'Radiation loss per turn U0 [keV] = %11.3e\n'%self.U0+'\n'
            s += 'fractional energy spread sige = %11.3e\n'%self.sige+'\n'
            s += ('-'*5*11+'\n'+5*'%11s')%('I1','I2','I3','I4','I5')+'\n'
            s += ('-'*5*11+'\n'+5*'%11.3e'+'\n')%(self.I[0],self.I[1],self.I[2],self.I[3],self.I[4])+'\n'
        if hasattr(self,'h1') or hasattr(self,'h2'):
            s += self.showh12()
        s += '\n\n'
        return s

    def rad(self):
        '''
        copy from S. Krinsky's code:
        Radiation integrals calculated according to Helm, Lee and Morton,  IEEE Trans. Nucl. Sc., NS-20, 1973, p900
        E = energy in GeV
        N = number of superperiods
        I = synchrotron integrals [I1,I2,I3,I4,I5]
        alphac = momentum compaction
        U0 = energy radiated per turn (keV)
        Jx,y,e = damping partition factors [Jx = 1-D, Jy = 1, Je = 2+D]
        tau= Damping time [taux,tauy,taue] (ms)
        sige = fractional energy spread
        emitx = horizonat emittance (nm)
        '''
        const = 55/(32*np.sqrt(3))
        re = 2.817940e-15 # m
        lam = 3.861592e-13 # m
        grel = 1956.95*self.E #E in GeV
        Ee = 0.510999 # MeV
        self.I = 5*[0.]
        for i, elem in enumerate(self.bl):
            if elem.__class__.__name__ == 'bend':
                L = elem.L
                K = elem.K1
                angle = elem.angle
                e1 = elem.e1
                e2 = elem.e2
                R = elem.R
                t1 = np.tan(e1)/R
                t2 = np.tan(e2)/R
                b0 = self.twx[0,i]
                a0 = self.twx[1,i]
                #g0 = self.twx[2,i]
                eta0 = self.dx[0,i]
                etap0 = self.dx[1,i]
                etap1 = etap0+eta0*t1
                a1 = a0-b0*t1
                g1 = (1+a1**2)/b0
                if K+1/R**2 > 0:
                    k = np.sqrt(K+1/R**2)
                    p = k*L
                    S = np.sin(p)
                    C = np.cos(p)
                    eta2 = eta0*C+etap1*S/k+(1-C)/(R*k**2)
                    av_eta = eta0*S/p+etap1*(1-C)/(p*k)+(p-S)/(p*k**2*R)
                    av2 = -K*av_eta/R+(eta0*t1+eta2*t2)/(2.*L*R)
                    av_H = g1*eta0**2+2*a1*eta0*etap1+b0*etap1**2 + \
                        (2.*angle)*(-(g1*eta0+a1*etap1)*(p-S)/(k*p**2)+(a1*eta0+b0*etap1)*(1-C)/p**2) + \
                        angle**2*(g1*(3.*p-4.*S+S*C)/(2.*k**2*p**3)-a1*(1.-C)**2/(k*p**3)+b0*(p-C*S)/(2.*p**3))
                elif K+1/R**2 < 0:
                    k = np.sqrt(-K-1/R**2)
                    p = k*L
                    S = np.sinh(p)
                    C = np.cosh(p)
                    eta2 = eta0*C+etap1*S/k+(1-C)/(-R*k**2)
                    av_eta = eta0*S/p+etap1*(1-C)/(-p*k)+(p-S)/(-p*k**2*R)
                    av2 = -K*av_eta/R+(eta0*t1+eta2*t2)/(2.*L*R)
                    av_H = g1*eta0**2+2*a1*eta0*etap1+b0*etap1**2 + \
                        (2.*angle)*(-(g1*eta0+a1*etap1)*(p-S)/(-k*p**2)+(a1*eta0+b0*etap1)*(1-C)*(-1.)/p**2) + \
                        angle**2*(g1*(3.*p-4.*S+S*C)/(2.*k**2*p**3)-a1*(1.-C)**2/(k*p**3)+b0*(p-C*S)/(-2.*p**3))
                else:
                    print('1/R**2+K1 == 0')
                    pass
                self.I[0] += L*av_eta/R
                self.I[1] += L/R**2
                self.I[2] += L/abs(R)**3
                self.I[3] += L*av_eta/R**3-2.*L*av2
                self.I[4] += L*av_H/abs(R)**3
        LSUP = self.s[-1]
        self.alphac = self.I[0]/LSUP
        self.U0 = self.N*666.66*re*Ee*grel**4*self.I[1]
        self.D = self.I[3]/self.I[1]
        self.Jx = 1-self.D
        self.Jy = 1.
        self.Je = 2+self.D
        self.tau0 = 3*LSUP*1e3/(re*csp*grel**3*self.I[1])
        self.taux = self.tau0/self.Jx
        self.tauy = self.tau0
        self.taue = self.tau0/self.Je
        self.sige = np.sqrt(const*lam*grel**2*self.I[2]/(2.*self.I[1]+self.I[3])) 
        self.emitx = const*lam*grel**2*(self.I[4]/(self.I[1]-self.I[3]))*1e9 #nm
        self.emity = self.emitx*self.kcouple/(1+self.kcouple) #nm
        self.sigx =[np.sqrt(self.betax[m]*self.emitx*1e-9) + 
                    abs(self.etax[m])*self.sige for m in range(len(self.s))]
        self.sigy =[np.sqrt(self.twy[0,m]*self.emity*1e-9) 
                    for m in range(len(self.s))]

    def chrom(self,cor=True):
        '''
        Calculates linear chromaticity dnu/(dE/E)
        original copy from S. Krinsky
        Simple, correct for quadrupoles and sextupoles
        Approximate for dipole with step profile
        Accurate for large bend radius and small bend angle per dipole
        when cor=False, it does not calculate chromaticity contribution
        from independent sextupoles
        '''
        #print('Warning: may NOT be good for small rings, dipole with gradient.')
        chx = 0.
        chy = 0.
        chx0 = 0.
        chy0 = 0.
        for i, elem in enumerate(self.bl):
            if elem.__class__.__name__ in ['drif','moni','aper']:
                continue
            s,w = gint(0.,elem.L)
            betax,alphax,gammax = [],[],[]
            betay,alphay,gammay = [],[],[]
            eta,etap=[],[]
            for j in range(8):
                mx,my,tx,ty = twmat(elem,s[j])
                wx = twisstrans(tx,self.twx[:,i])
                wy = twisstrans(ty,self.twy[:,i])
                dx = mx*self.dx[:,i]
                betax.append(wx[0,0])
                alphax.append(wx[1,0])
                gammax.append(wx[2,0])
                betay.append(wy[0,0])
                alphay.append(wy[1,0])
                gammay.append(wy[2,0])
                eta.append(dx[0,0])
                etap.append(dx[1,0])
            betax,alphax,gammax = np.array(betax),np.array(alphax),np.array(gammax)
            betay,alphay,gammay = np.array(betay),np.array(alphay),np.array(gammay)
            eta,etap = np.array(eta),np.array(etap)
            if elem.__class__.__name__ == 'quad':
                dchx = -elem.K1*sum(w*betax)
                dchy =  elem.K1*sum(w*betay)
                chx += dchx
                chy += dchy
                chx0 += dchx
                chy0 += dchy
            elif elem.__class__.__name__ == 'bend':
                h = elem.angle/elem.L
                if elem.K1 == 0.:
                    dchx = -h**2*sum(w*betax)-2*h*sum(w*etap*alphax)+h*sum(w*eta*gammax)
                    dchy = h*sum(w*eta*gammay)
                else:
                    dchx = -(elem.K1+h**2)*sum(w*betax)-2*elem.K1*h*sum(w*eta*betax) - \
                        2*h*sum(w*etap*alphax)+h*sum(w*eta*gammax)
                    dchy = elem.K1*sum(w*betay)-h*elem.K1*sum(w*eta*betay)+h*sum(w*eta*gammay)
                if hasattr(elem,'K2'):
                    dchx +=  elem.K2*sum(w*eta*betax)
                    dchy += -elem.K2*sum(w*eta*betay)
                b0x = self.twx[0,i]
                b0y = self.twy[0,i]
                b1x = self.twx[0,i+1]
                b1y = self.twy[0,i+1]
                dchx +=  h*b0x*np.tan(elem.e1)+h*b1x*np.tan(elem.e2)
                dchy += -h*b0y*np.tan(elem.e1)-h*b1y*np.tan(elem.e2)
                chx += dchx
                chy += dchy
                chx0 += dchx
                chy0 += dchy
            elif elem.__class__.__name__ == 'sext' and cor:
                if elem.L == 0.:
                    dchx =  elem.K2*self.dx[0,i]*self.twx[0,i]
                    dchy = -elem.K2*self.dx[0,i]*self.twy[0,i]
                else:
                    dchx = elem.K2*sum(w*eta*betax)
                    dchy =-elem.K2*sum(w*eta*betay)
                chx += dchx
                chy += dchy
        self.chx = chx/(4*np.pi)*self.N
        self.chy = chy/(4*np.pi)*self.N
        self.chx0 = chx0/(4*np.pi)*self.N
        self.chy0 = chy0/(4*np.pi)*self.N


    def cchrom(self,var,ch=[0,0],minSextValue=False):
        '''
        use specified sexts to correct chromaticity
        if minSextValue is True, try to minimize the 
        sum of absolute sextupole strengthes
        '''
        sexts = [[v,'K2'] for v in var]
        if minSextValue:
            self.svabs = 0
            for s in sexts: self.svabs += np.abs(s[0].K2)
            con = [['chx',float(ch[0])],['chy',float(ch[1])],['svabs',0.]]
            wgh = [1.0,1.0,0.5]
        else:
            con = [['chx',float(ch[0])],['chy',float(ch[1])]]
            wgh = [1.0,1.0]
        optm(self,sexts,con,wgh)
        if hasattr(self, 'svabs'):
            delattr(self, 'svabs')
        self.chrom()

    def showh12(self):
        '''
        get a string to display h1 and h2
        '''
        str0 = ''
        if hasattr(self,'h1'):
            str0 += '\n=* First order driving terms *=\n'
            for k in self.h1.keys():
                rr = float(np.real(self.h1[k]))
                ii = float(np.imag(self.h1[k]))
                str0 += '%s = %15.6e %+15.6ej\n'%(k,rr,ii)
        if hasattr(self,'h2'):
            str0 += '\n=* Second order driving terms *=\n'
            for k in self.h2.keys():
                rr = float(np.real(self.h2[k]))
                ii = float(np.imag(self.h2[k]))
                str0 += '%s = %15.6e %+15.6ej\n'%(k,rr,ii)
        return str0

    def plth12(self,save=False):
        '''
        plot h1 and h2 in polar coordinate
        '''
        plt.figure()
        if hasattr(self,'h1'):
            plt.subplot(1,2,1)
            lgd = []
            for k in self.h1.keys():
                plt.plot([0,np.real(self.h1[k])],[0,np.imag(self.h1[k])],'-+',linewidth=2)
                lgd.append(k)
            plt.axis('equal')
            plt.legend(lgd)
        if hasattr(self,'h2'):
            plt.subplot(1,2,2)
            lgd = []
            for k in self.h2.keys():
                plt.plot([0,np.real(self.h2[k])],[0,np.imag(self.h2[k])],'-+',linewidth=2)
                lgd.append(k)
            plt.axis('equal')
            plt.legend(lgd)
        if save:
            plt.savefig('h12.eps')
            plt.savefig('h12.png')
        plt.show()

    def twmid(self,i):
        '''
        twiss parameter in the middle of the i-th elem
        '''
        ei = self.bl[i]
        mx,my,tx,ty = twmat(ei,ei.L/2)
        twx = twisstrans(tx,self.twx[:,i])
        twy = twisstrans(ty,self.twy[:,i])
        dx = mx*self.dx[:,i]
        bxi,byi = twx[0,0],twy[0,0]
        dxi = dx[0,0]
        dpx = phasetrans(mx,self.twx[:,i])
        dpy = phasetrans(my,self.twy[:,i])
        muxi,muyi = self.mux[i]+dpx,self.muy[i]+dpy
        return bxi,byi,dxi,muxi*2*np.pi,muyi*2*np.pi

    def geth1(self):
        '''
        non-linear driving terms
        '''
        h11001 = complex(0,0)
        h00111 = complex(0,0)
        h20001 = complex(0,0)
        h20001 = complex(0,0)
        h00201 = complex(0,0)
        h10002 = complex(0,0)

        h21000 = complex(0,0)
        h30000 = complex(0,0)
        h10110 = complex(0,0)
        h10020 = complex(0,0)
        h10200 = complex(0,0)


        for i,ei in enumerate(self.bl):
            if ei.__class__.__name__ == 'sext':
                bxi,byi,dxi,muxi,muyi = self.twmid(i)
                K2Li = ei.K2*ei.L/2
                h11001 -= 2*K2Li*dxi*bxi 
                h00111 += 2*K2Li*dxi*byi 
                h20001 -= 2*K2Li*dxi*bxi*np.exp(2j*muxi)
                h00201 += 2*K2Li*dxi*byi*np.exp(2j*muyi)
                h10002 -= K2Li*dxi**2*np.sqrt(bxi)*np.exp(1j*muxi)

                h21000 -= K2Li*bxi**(3./2)*np.exp(1j*muxi)
                h30000 -= K2Li*bxi**(3./2)*np.exp(3j*muxi)
                h10110 += K2Li*np.sqrt(bxi)*byi*np.exp(1j*muxi)
                h10020 += K2Li*np.sqrt(bxi)*byi*np.exp(1j*(muxi-2*muyi))
                h10200 += K2Li*np.sqrt(bxi)*byi*np.exp(1j*(muxi+2*muyi))
                
            if ei.__class__.__name__ == 'quad':
                bxi,byi,dxi,muxi,muyi = self.twmid(i)
                K1Li = ei.K1*ei.L
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)

            if ei.__class__.__name__ == 'bend':
                #entrance fringe
                K1Li = -np.tan(ei.e1)/ei.R
                bxi,byi,dxi,muxi,muyi = self.twx[0,i],self.twy[0,i],self.dx[0,i],self.mux[i],self.muy[i]
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)
                #body
                bxi,byi,dxi,muxi,muyi = self.twmid(i)
                #horizontal
                K1Li = (ei.K1+1/ei.R**2)*ei.L
                h11001 += K1Li*bxi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)
                #vertical
                K1Li = ei.K1*ei.L
                h00111 -= K1Li*byi
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                #exit fringe
                K1Li = -np.tan(ei.e2)/ei.R
                bxi,byi,dxi,muxi,muyi = self.twx[0,i+1],self.twy[0,i+1],self.dx[0,i+1],self.mux[i+1],self.muy[i+1]
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)

        self.h1 = {'h11001':h11001/4,'h00111':h00111/4,'h20001':h20001/8,'h00201':h00201/8,
                   'h10002':h10002/2,'h21000':h21000/8,'h30000':h30000/24,'h10110':h10110/4,
                   'h10020':h10020/8,'h10200':h10200/8}

    def geth2(self):
        '''
        non-linear driving terms
        ref: Second-order driving terms due to sextupoles and chromatic effects of quadrupoles
        AOP-TN-2009-020
        Chun-xi Wang
        '''
        h22000 = complex(0,0)
        h31000 = complex(0,0)
        h11110 = complex(0,0)
        h11200 = complex(0,0)
        h40000 = complex(0,0)
        h20020 = complex(0,0)
        h20110 = complex(0,0)
        h20200 = complex(0,0)
        h00220 = complex(0,0)
        h00310 = complex(0,0)
        h00400 = complex(0,0)

        h21001 = complex(0,0)
        h30001 = complex(0,0)
        h10021 = complex(0,0)
        h10111 = complex(0,0)
        h10201 = complex(0,0)
        h11002 = complex(0,0)
        h20002 = complex(0,0)
        h00112 = complex(0,0)
        h00202 = complex(0,0)
        h10003 = complex(0,0)
        h00004 = complex(0,0)
        
        for i,ei in enumerate(self.bl):
            if ei.__class__.__name__ == 'sext':
                bxi,byi,dxi,muxi,muyi = self.twmid(i)
                K2Li = ei.K2*ei.L/2.
                for j,ej in enumerate(self.bl):
                    if ej.__class__.__name__ == 'sext':
                        if i != j:
                            bxj,byj,dxj,muxj,muyj = self.twmid(j)
                            K2Lj = ej.K2*ej.L/2.
                            d22000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*(np.exp(3j*(muxi-muxj))+\
                                                                      3*np.exp(1j*(muxi-muxj)))
                            d31000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*np.exp(1j*(3*muxi-muxj))
                            d11110 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-muxj))-\
                                                                          np.exp(1j*(muxi-muxj)))+\
                                                                     byj*(np.exp(1j*(muxi-muxj+2*muyi-2*muyj))+\
                                                                          np.exp(-1j*(muxi-muxj-2*muyi+2*muyj))))
                            d11200 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-muxj-2*muyi))-\
                                                                          np.exp(1j*(muxi-muxj+2*muyi)))+\
                                                                   2*byj*(np.exp(1j*(muxi-muxj+2*muyi))+\
                                                                          np.exp(-1j*(muxi-muxj-2*muyj))))
                            d40000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*np.exp(1j*(3*muxi+muxj))
                            d20020 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*np.exp(-1j*(muxi-3*muxj+2*muyi))-\
                                                             (bxj+4*byj)*np.exp(1j*(muxi+muxj-2*muyi)))
                            d20110 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-3*muxj))-np.exp(1j*(muxi+muxj)))+\
                                                                    2*byj*np.exp(1j*(muxi+muxj+2*muyi-2*muyj)))
                            d20200 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*np.exp(-1j*(muxi-3*muxj-2*muyi))-\
                                                             (bxj-4*byj)*np.exp(1j*(muxi+muxj+2*muyi)))
                            d00220 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*(np.exp(1j*(muxi-muxj+2*muyi-2*muyj))+\
                                                                       4*np.exp(1j*(muxi-muxj))-\
                                                                         np.exp(-1j*(muxi-muxj-2*muyi+2*muyj)))
                            d00310 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*(np.exp(1j*(muxi-muxj+2*muyi))-\
                                                                         np.exp(-1j*(muxi-muxj-2*muyi)))
                            d00400 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*np.exp(1j*(muxi-muxj+2*muyi+2*muyj))
                            # --- geometric/chromatic
                            d21001 = -1j/16*K2Li*K2Lj*bxi*bxj**(3./2)*dxi*(np.exp(1j*muxj)-\
                                                                         2*np.exp(1j*(2*muxi-muxj))+\
                                                                           np.exp(-1j*(2*muxi-3*muxj)))
                            d30001 = -1j/16*K2Li*K2Lj*bxi*bxj**(3./2)*dxi*(np.exp(3j*muxj)-\
                                                                           np.exp(1j*(2*muxi+muxj)))
                            d10021 = 1j/16*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*(muxj-2*muyj))-\
                                                                               np.exp(1j*(2*muxi-muxj-2*muyj)))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj-2*muyi))-\
                                                                              np.exp(1j*(muxj-2*muyj)))
                            d10111 = 1j/8*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*muxj)-\
                                                                              np.exp(1j*(2*muxi-muxj)))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj-2*muyi+2*muyj))-\
                                                                              np.exp(1j*(muxj+2*muyi-2*muyj)))
                            d10201 = 1j/16*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*(muxj+2*muyj))-\
                                                                               np.exp(1j*(2*muxi-muxj+2*muyj)))+\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj+2*muyi))-\
                                                                              np.exp(1j*(muxj+2*muyj)))
                            d11002 = 1j/4*K2Li*K2Lj*bxi*bxj*dxi*dxj*np.exp(2j*(muxi-muxj))+\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi**2*(np.exp(1j*(muxi-muxj))-\
                                                                                     np.exp(-1j*(muxi-muxj)))
                            d20002 = 1j/4*K2Li*K2Lj*bxi*bxj*dxi*dxj*np.exp(2j*muxi)+\
                                     1j/16*K2Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi**2*(np.exp(1j*(muxi+muxj))-\
                                                                                     np.exp(-1j*(muxi-3*muxj)))
                            d00112 = 1j/4*K2Li*K2Lj*byi*byj*dxi*dxj*np.exp(2j*(muyi-muyj))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi**2*(np.exp(1j*(muxi-muxj))-\
                                                                                 np.exp(-1j*(muxi-muxj)))
                            d00202 = 1j/4*K2Li*K2Lj*byi*byj*dxi*dxj*np.exp(2j*muyi)-\
                                     1j/16*K2Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi**2*(np.exp(1j*(muxi-muxj+2*muyj))-\
                                                                                  np.exp(-1j*(muxi-muxj-2*muyj)))
                            d10003 = 1j/4*K2Li*K2Lj*np.sqrt(bxi)*bxj*dxi**2*dxj*(np.exp(1j*muxi)-\
                                                                                 np.exp(-1j*(muxi-2*muxj)))
                            d00004 = 1j/4*K2Li*K2Lj*np.sqrt(bxi*bxj)*dxi**2*dxj**2*np.exp(1j*(muxi-muxj))
                                                                              
                            if j > i:
                                h22000 += d22000
                                h31000 += d31000
                                h11110 += d11110
                                h11200 += d11200
                                h40000 += d40000
                                h20020 += d20020
                                h20110 += d20110
                                h20200 += d20200
                                h00220 += d00220 
                                h00310 += d00310
                                h00400 += d00400
                                h21001 += d21001
                                h30001 += d30001
                                h10021 += d10021
                                h10111 += d10111
                                h10201 += d10201
                                h11002 += d11002
                                h20002 += d20002
                                h00112 += d00112
                                h00202 += d00202
                                h10003 += d10003
                                h00004 += d00004

                            elif j < i:
                                h22000 -= d22000
                                h31000 -= d31000
                                h11110 -= d11110
                                h11200 -= d11200
                                h40000 -= d40000
                                h20020 -= d20020
                                h20110 -= d20110
                                h20200 -= d20200
                                h00220 -= d00220 
                                h00310 -= d00310
                                h00400 -= d00400
                                h21001 -= d21001
                                h30001 -= d30001
                                h10021 -= d10021
                                h10111 -= d10111
                                h10201 -= d10201
                                h11002 -= d11002
                                h20002 -= d20002
                                h00112 -= d00112
                                h00202 -= d00202
                                h10003 -= d10003
                                h00004 -= d00004

                    if ej.__class__.__name__ == 'quad':
                        bxj,byj,dxj,muxj,muyj = self.twmid(j)
                        K1Lj = ej.K1*ej.L
                        d21001 = -1j/32*K2Li*K1Lj*bxi**(3./2)*bxj*(np.exp(1j*muxi)+\
                                                                   np.exp(1j*(3*muxi-2*muxj))-\
                                                                 2*np.exp(-1j*(muxi-2*muxj)))
                        d30001 = -1j/32*K2Li*K1Lj*bxi**(3./2)*bxj*(np.exp(3j*muxi)-\
                                                                   np.exp(1j*(muxi+2*muxj)))
                        d10021 = 1j/32*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*(muxi-2*muyi))-\
                                                                       np.exp(-1j*(muxi-2*muxj+2*muyi)))+\
                                 1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi-2*muyi))-\
                                                                       np.exp(1j*(muxi-2*muyj)))
                        d10111 = 1j/16*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*muxi)-\
                                                                       np.exp(-1j*(muxi-2*muxj)))+\
                                 1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi-2*muyi+2*muyj))-\
                                                                       np.exp(1j*(muxi+2*muyi-2*muyj)))
                        d10201 = 1j/32*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*(muxi+2*muyi))-\
                                                                       np.exp(-1j*(muxi-2*muxj-2*muyi)))-\
                                 1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi+2*muyi))-\
                                                                       np.exp(1j*(muxi+2*muyj)))
                        d11002 = -1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*(muxi-muxj))+\
                                  1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(-2j*(muxi-muxj))
                        d20002 = -1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*muxi)+\
                                  1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*muxj)
                        d00112 = -1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*(muyi-muyj))+\
                                  1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(-2j*(muyi-muyj))
                        d00202 = -1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*muyi)+\
                                  1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*muyj)
                        d10003 =  1j/4*K2Li*K1Lj*bxi*np.sqrt(bxj)*dxi*dxj*(np.exp(1j*muxj)-\
                                                                           np.exp(1j*(2*muxi-muxj)))-\
                                  1j/8*K2Li*K1Lj*np.sqrt(bxi)*bxj*dxi**2*(np.exp(1j*muxi)-\
                                                                          np.exp(-1j*(muxi-2*muxj)))
                        d00004 = -1j/4*K2Li*K1Lj*np.sqrt(bxi*bxj)*dxi**2*dxj*np.exp(1j*(muxi-muxj))+\
                                  1j/4*K2Li*K1Lj*np.sqrt(bxi*bxj)*dxi**2*dxj*np.exp(-1j*(muxi-muxj))

                        if j > i:
                            h21001 += d21001
                            h30001 += d30001
                            h10021 += d10021
                            h10111 += d10111
                            h10201 += d10201
                            h11002 += d11002
                            h20002 += d20002
                            h00112 += d00112
                            h00202 += d00202
                            h10003 += d10003
                            h00004 += d00004

                        elif j < i:
                            h21001 -= d21001
                            h30001 -= d30001
                            h10021 -= d10021
                            h10111 -= d10111
                            h10201 -= d10201
                            h11002 -= d11002
                            h20002 -= d20002
                            h00112 -= d00112
                            h00202 -= d00202
                            h10003 -= d10003
                            h00004 -= d00004
                        
            if ei.__class__.__name__ == 'quad':
                bxi,byi,dxi,muxi,muyi = self.twmid(i)
                K1Li = ei.K1*ei.L
                for j,ej in enumerate(self.bl):
                    if ej.__class__.__name__ == 'sext':
                        bxj,byj,dxj,muxj,muyj = self.twmid(j)
                        K2Lj = ej.K2*ej.L/2.
                        d11002 = -1j/8*K1Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi*(np.exp(1j*(muxi-muxj))-\
                                                                               np.exp(-1j*(muxi-muxj)))
                        d20002 = -1j/16*K1Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi*(np.exp(1j*(muxi+muxj))-\
                                                                                np.exp(-1j*(muxi-3*muxj)))
                        d00112 = 1j/8*K1Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi*(np.exp(1j*(muxi-muxj))-\
                                                                          np.exp(-1j*(muxi-muxj)))
                        d00202 = 1j/16*K1Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi*(np.exp(1j*(muxi-muxj+2*muyj))-\
                                                                           np.exp(-1j*(muxi-muxj-2*muyj)))

                        if j > i:
                            h11002 += d11002
                            h20002 += d20002
                            h00112 += d00112
                            h00202 += d00202

                        elif j < i:
                            h11002 -= d11002
                            h20002 -= d20002
                            h00112 -= d00112
                            h00202 -= d00202

                    if ej.__class__.__name__ == 'quad':
                        if i != j:
                            bxj,byj,dxj,muxj,muyj = self.twmid(j)
                            K1Lj = ej.K1*ej.L
                            d11002 = 1j/16*K1Li*K1Lj*bxi*bxj*np.exp(2j*(muxi-muxj))
                            d20002 = 1j/16*K1Li*K1Lj*bxi*bxj*np.exp(2j*muxi)
                            d00112 = 1j/16*K1Li*K1Lj*byi*byj*np.exp(2j*(muyi-muyj))
                            d00202 = 1j/16*K1Li*K1Lj*byi*byj*np.exp(2j*muyi)
                            d10003 = 1j/8*K1Li*K1Lj*np.sqrt(bxi)*bxj*dxi*(np.exp(1j*muxi)-\
                                                                          np.exp(-1j*(muxi-2*muxj)))
                            d00004 = 1j/4*K1Li*K1Lj*np.sqrt(bxi*bxj)*dxi*dxj*np.exp(1j*(muxi-muxj))
                        if j > i:
                            h11002 += d11002
                            h20002 += d20002
                            h00112 += d00112
                            h00202 += d00202
                            h10003 += d10003
                            h00004 += d00004
                        elif j < i:
                            h11002 -= d11002
                            h20002 -= d20002
                            h00112 -= d00112
                            h00202 -= d00202
                            h10003 -= d10003
                            h00004 -= d00004

        self.h2 = {'h22000':h22000*1j/64,'h31000':h31000*1j/32,'h11110':h11110*1j/16,'h11200':h11200*1j/32,'h40000':h40000*1j/64,
                   'h20020':h20020*1j/64,'h20110':h20110*1j/32,'h20200':h20200*1j/64,'h00220':h00220*1j/64,'h00310':h00310*1j/32,
                   'h00400':h00400*1j/64,'h21001':h21001,'h30001':h30001,'h10021':h10021,'h10111':h10111,'h10201':h10201,
                   'h11002':h11002,'h20002':h20002,'h00112':h00112,'h00202':h00202,'h10003':h10003,'h00004':h00004}

    def getOrm(self,bpms=None,cors=None):
        '''
        get Obit Response Matrix to correctors from optics
        so far, no coupling included
        '''
        if not bpms:
            bpms = self.getElements('moni')
        if not cors:
            cors = self.getElements('drif','C')
        ormx = np.zeros((len(bpms),len(cors)))
        ormy = np.zeros((len(bpms),len(cors)))
        for m,cor in enumerate(cors):
            for n,bpm in enumerate(bpms):
                ic = self.bl.index(cor)
                ib = self.bl.index(bpm)
                sbx = np.sqrt(self.betax[ic]*self.betax[ib])
                sby = np.sqrt(self.betay[ic]*self.betay[ib])
                cpx = np.cos(abs(self.mux[ic]-self.mux[ib])*2*np.pi-np.pi*self.nux)
                cpy = np.cos(abs(self.muy[ic]-self.muy[ib])*2*np.pi-np.pi*self.nuy)
                ormx[n,m] = sbx*cpx
                ormy[n,m] = sby*cpy
        ormx /= 2*np.sin(np.pi*self.nux)
        ormy /= 2*np.sin(np.pi*self.nuy)
        self.ormx,self.ormy = ormx,ormy
 
    def getBBrmg(self,monis=None,quads=None,tune=False,
                tunewgh=100,dk=0.001):
        '''
        get Beta-Beat Response Matrix to quads by varying quadrupole strengths

        monis: observation element, where beta-function measured 
        quads: quads can be varied to measurement RM 

        If tune is True, BBrm includes tune(phase)-shift with quads.
        Because tune-shifts with quads are small, their weights are chosen
        100 by default, therefore the cooresponding goal function needs to
        have same weights
        '''
        if not monis:
            monis = self.getElements('quad')
        if not quads:
            quads = self.getElements('quad')
        if tune:
            bbrmx = np.zeros((len(monis)+1,len(quads)))
            bbrmy = np.zeros((len(monis)+1,len(quads)))
        else:
            bbrmx = np.zeros((len(monis),len(quads)))
            bbrmy = np.zeros((len(monis),len(quads)))
        mi = [self.bl.index(q) for q in monis]
        bx0 = np.copy(self.betax[mi])
        by0 = np.copy(self.betay[mi])
        if tune:
            bx0 = np.append(bx0,self.mux[-1]*100)
            by0 = np.append(by0,self.muy[-1]*100)
        for m,qm in enumerate(quads):
            print('%4i/%4i: %s'%(m,len(quads),qm.name))
            qm.put('K1',qm.K1+dk)
            self.update()
            bx1 = np.copy(self.betax[mi])
            by1 = np.copy(self.betay[mi])
            if tune:
                bx1 = np.append(bx1,self.mux[-1]*100)
                by1 = np.append(by1,self.muy[-1]*100)
            bbrmx[:,m] = (bx1-bx0)/dk
            bbrmy[:,m] = (by1-by0)/dk
            qm.put('K1',qm.K1-dk)
        self.bbrmx,self.bbrmy = bbrmx,bbrmy
        self.update()

    def getBBrm1(self,monis=None,quads=None):
        '''
        --- need to work on it!
        get Beta-Beat Response Matrix to quads from optics
        monis: observation points, where beta-function measured
        quads: quads can be varied to measurement RM
        '''
        print('=== Do NOT use it, bugs needed to be fixed! ===')
        print('\n calculate Beta-beat Response Matrix from optics')
        return
        if not monis:
            monis = self.getElements('quad')
        if not quads:
            quads = self.getElements('quad')
        bbrmx = np.zeros((len(monis),len(quads)))
        bbrmy = np.zeros((len(monis),len(quads)))
        mi = [self.bl.index(q) for q in monis]
        bx0 = np.copy(self.betax[mi])
        by0 = np.copy(self.betay[mi])
        for m,qm in enumerate(quads):
            for n,mn in enumerate(monis):
                iq = self.bl.index(qm)
                im = self.bl.index(mn)
                sbx = (self.betax[iq]+self.betax[iq+1])/2* \
                      (self.betax[im]+self.betax[im+1])/2*qm.L
                sby = (self.betay[iq]+self.betay[iq+1])/2* \
                      (self.betay[im]+self.betay[im+1])*2*qm.L
                cpx = np.cos(2*np.pi*(2*self.mux[im]-2*self.mux[iq]+self.nux))
                cpy = np.cos(2*np.pi*(2*self.muy[im]-2*self.muy[iq]+self.nuy))
                bbrmx[n,m] = sbx*cpx
                bbrmy[n,m] = sby*cpy
        bbrmx /= 2*np.sin(2*np.pi*self.nux)
        bbrmy /=-2*np.sin(2*np.pi*self.nuy)
        self.bbrmx,self.bbrmy = bbrmx,bbrmy
        self.update()
 
    def finddyap(self,xmin=-3e-2, xmax=3e-2, ymin=0, ymax=1.5e-2,
                 nx=61, ny=16, dp=0, nturn=512):
        '''
        find dynamic aperture with matrix tracking and thin-len kicks
        '''
        x = np.linspace(xmin,xmax,nx)
        y = np.linspace(ymin,ymax,ny)
        xgrid,ygrid = np.meshgrid(x,y)
        xin = np.zeros((7,nx*ny))
        xin[-1] = np.arange(nx*ny)
        xin[0] = xgrid.flatten()
        xin[2] = ygrid.flatten()
        xin[5] = dp
        xout = self.matrack(xin,nturn=nturn)
        dyap = np.zeros(nx*ny)
        dyap[xout[-1].astype(int)] = 1
        dyap = dyap.reshape(xgrid.shape)
        self.dyap = {'xgrid':xgrid,'ygrid':ygrid,'dyap':dyap,'dp':dp}

    def pltdyap(self):
        '''
        plot dynamic aperture
        '''
        if self.dyap:
            plt.contourf(self.dyap['xgrid'],self.dyap['ygrid'],self.dyap['dyap'])
            plt.xlabel('x[m]')
            plt.ylabel('y[m]')
            plt.title('Dynamic aperture for dp/p = %.3f'%self.dyap['dp'])
            plt.show()
        else:
            print('dynamic aperture has been not calculated yet.\n')

    def switchoffID(self,ids=None):
        '''
        switch off insertion devices,
        change it into drift
        '''
        print('\n === Switched-off IDs can NOT be swithed on ===')
        if not ids:
            ids = self.getElements('kmap')
        for ie in ids:
            ii = self.bl.index(ie)
            rep = drif(ie.name,L=ie.L)
            self.bl[ii] = rep
        self.update()
        

def svdcor(x,rm,rcond=1e-6):
    '''
    use svd to correct error consequence by response matrix rm
    usage: svd(x,rm, rcond=0.01)

    parameters:
    x: observed error consequence needs to be corrected, 
       like closed orbit distortion, beta-beat, etc
    rm: response matrix between observed signals and correctors
    rcond: drop small singual values off by comparing 
           with the largest one (0th)
           if si/s0 < rcond, don't use it for correction
    returns:
    dk:  needed corrector strength
    xr:  expectation after correction
    '''
    u,s,v = np.linalg.svd(rm)
    dk = np.zeros(v.shape[0], dtype=np.float)
    for m in range(len(s)):
        if s[m]/s[0] >= rcond:
            dk -= v[m,:]/s[m]*np.dot(x, u[:,m])
        else:
            break
    xr = np.mat(x).reshape(-1,1)-np.mat(rm)*np.mat(dk).reshape(-1,1)
    return dk,np.array(xr)[:,0]


def micado(x,rm,n=1,verbose=False):
    '''
    use  MICADO to correct error consequence by response matrix rm
    usage: micado(x,rm, n=1)

    parameters:
    x:       observed error consequence needs to be corrected, 
             like closed orbit distortion, beta-beat, etc
    rm:      response matrix between observed signals and correctors
    n:       number of iterations
    verbose: if true, print out details 

    returns:
    dk: needed corrector strength
    xr: expectation after corrector 
    '''
    k,c,xr = [],[],[]
    if verbose:
        print('%9s%9s%9s%9s'%('#iter','#corr','dk','rsd_err'))
        print(36*'-')
    for n in range(n):
        # --- a[i] is the most effective coefficient for i-th corrector
        # --- xres is residual after ampplying a
        # --- xressum is sum of residual
        a = np.zeros(rm.shape[1])
        xres = np.zeros_like(rm)
        xressum = np.zeros_like(a)
        for i in range(rm.shape[1]):
            a[i] = sum(x*rm[:,i])/sum(rm[:,i]*rm[:,i])
            xres[:,i] = x-a[i]*rm[:,i]
            xressum[i] = sum(xres[:,i]*xres[:,i])
        ki = np.argmin(xressum) # No. 1 effective corr
        x = xres[:,ki]          # residual become goal to be corrected
        k.append(ki)            # corrector index
        c.append(a[ki])         # corrector value
        xr.append(xressum[ki])  # residual
        if verbose:
            print('%9d%9d%9.3f%9.3f'%(n,ki,a[ki],xressum[ki]))
    dk = np.zeros(rm.shape[1])
    # --- sum over 
    for i,ki in enumerate(k):
        dk[ki] -= c[i]
    return dk,x


def gint(a,b):
    '''
    Eight point Gaussian Integration over interval a<x<b
    x,w = gint(a,b)
    '''
    y = np.array([-0.960289856497536,-0.796666477413627,-0.525532409916329,
                  -0.183434642495650, 0.183434642495650, 0.525532409916329,
                   0.796666477413627, 0.960289856497536])
    v = np.array([ 0.101228536290376, 0.222381034453374, 0.313706645877887,
                   0.362683783378362, 0.362683783378362, 0.313706645877887,
                   0.222381034453374, 0.101228536290376])
    x = 0.5*(b-a)*y+0.5*(b+a)
    w = 0.5*(b-a)*v
    return x,w


def period(bl):
    '''
    Determines the twiss functions and dispersion 
    at a single location of a ring
    wx = matrix([[betax],[alfax],[gamax]])
    wy = np.mat([[betay],[alfay],[gamay]])
    R is the single turn transport matrix
    dx=np.mat([[[etax],[etaxp]])
    isstable,R,wx,wy,dx = period(bl)
    isstable = True, if bl has a stable optics solution
    otherwise Flase.
    '''
    R = np.mat(np.eye(6))
    for elem in bl:
         R = elem.tm*R
    if abs(R[0,0]+R[1,1]) >= 2 or abs(R[2,2]+R[3,3]) >= 2:
        return False,R,None,None,None
    mux,betax,alfax,gamax = matrix2twiss(R,plane='x')
    muy,betay,alfay,gamay = matrix2twiss(R,plane='y')
    dx = matrix2disp(R,plane='x')
    wx = np.mat([betax,alfax,gamax]).transpose()
    wy = np.mat([betay,alfay,gamay]).transpose()
    return True,R,wx,wy,dx


def twmat(elem,s):
    '''
    Computes the transport and twiss matrix for transport
    through distance s in element elem.
    usage: mat = twmat(elem,s)
    '''
    if elem.__class__.__name__=='drif' or \
       elem.__class__.__name__=='sext' or \
       elem.__class__.__name__=='kick' or \
       elem.__class__.__name__=='aper':
        fe = drif(L=s)
    elif elem.__class__.__name__ == 'quad':
        fe = quad(L=s,K1=elem.K1)
    elif elem.__class__.__name__ == 'bend':
        if s == elem.L:
            fe = bend(L=elem.L,K1=elem.K1,angle=elem.angle,
                      e1=elem.e1,e2=elem.e2)
        elif s < elem.L:
            fe = bend(L=s,K1=elem.K1,angle=elem.angle*s/elem.L,e1=elem.e1)
    elif elem.__class__.__name__ == 'kmap':
        fe = kmap(L=s,kmap1fn=elem.kmap1fn,kmap2fn=elem.kmap2fn,E=elem.E)
    elif elem.__class__.__name__ == 'wigg':
        fe = wigg(L=s,Bw=elem.Bw,E=elem.E)
    else:
        raise RuntimeError('unknown type %s'%elem.name)
    mx = np.take(np.take(fe.tm,[0,1,5],axis=0),[0,1,5],axis=1)
    my = np.take(np.take(fe.tm,[2,3,5],axis=0),[2,3,5],axis=1)
    return mx,my,fe.tx,fe.ty


def twisstrans(twissmatrix,twiss0):
    '''
    get tiwss functions by twiss transport matrix
    '''
    return twissmatrix*twiss0


def phasetrans(m,tw0):
    '''
    get phase advance by transport matrix
    '''
    dph = np.arctan(m[0,1]/(m[0,0]*tw0[0,0]-m[0,1]*tw0[1,0]))/(2*np.pi)
    if dph < 0.:
        dph += 0.5
    return dph


def trans2twiss(a):
    '''
    calculate twiss matrix from transport matrix
    input (a) is a 2x2 coordinate transport matrix
    return a 3x3 twiss transport matrix [beta, alfa, gama]
    '''
    return np.mat([[a[0,0]**2,-2*a[0,0]*a[0,1],a[0,1]**2],
                   [-a[1,0]*a[0,0],1+2*a[0,1]*a[1,0],-a[0,1]*a[1,1]],
                   [a[1,0]**2,-2*a[1,1]*a[1,0],a[1,1]**2]])


def matrix2twiss(R,plane='x'):
    '''
    get fractional mu and Twiss from one turn transport matrix
    '''
    if plane == 'x':
        r = np.take(np.take(R,[0,1,5],axis=0),[0,1,5],axis=1)
    elif plane == 'y':
        r = np.take(np.take(R,[2,3,5],axis=0),[2,3,5],axis=1)
    else:
        print('plane must be \'x\' or \'y\'')
        return
    if abs(r[0,0]+r[1,1]) >= 2:
        print('det(M) >= 2: no stable solution in %s plane'%plane)
        return
    mu = np.arccos((r[0,0]+r[1,1])/2)/(2*np.pi)
    if r[0,1] < 0:
        mu = 1-mu
    s = np.sin(2*np.pi*mu)
    beta = r[0,1]/s
    alfa = (r[0,0]-r[1,1])/(2*s)
    gama = -r[1,0]/s
    return mu,beta,alfa,gama
    

def matrix2disp(R,plane='x'):
    '''
    get dispersion from one turn transport matrix
    '''
    if plane == 'x':
        r = np.take(np.take(R,[0,1,5],axis=0),[0,1,5],axis=1)
    elif plane == 'y':
        r = np.take(np.take(R,[2,3,5],axis=0),[2,3,5],axis=1)
    else:
        print('plane must be \'x\' or \'y\'')
        return
    det = (r[0,0]-1)*(r[1,1]-1)-r[0,1]*r[1,0]
    mm = np.mat([[r[1,1]-1,-r[0,1]],[-r[1,0],r[0,0]-1]])/det
    xx = -r[0:2,2]
    return np.append(mm*xx,np.mat([[1]]),axis=0)


def txt2latt(fn):
    '''
    read nls2-ii control-system lattice format
    '''
    fid = open(fn,'r')
    a = fid.readlines()
    fid.close()
    bl = []
    for le in a[3:]:
        ele = le.split()
        if ele[1] =='DW':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'IVU':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'EPU':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'SQ_TRIM':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'BPM':
            t = moni(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMD':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'MARK':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMY':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMX':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'FTRIM':
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'SEXT':
            t = sext(ele[0],L=float(ele[2]),K2=float(ele[5]))
        elif ele[1] == 'DIPOLE':
            t = bend(ele[0],L=float(ele[2]),
                     angle=float(ele[6]),e1=float(ele[6])/2,e2=float(ele[6])/2)
        elif ele[1] == 'QUAD':
            t = quad(ele[0],L=float(ele[2]),K1=float(ele[4]))
        elif ele[1] == 'DRIF':
            t = drif(ele[0],L=float(ele[2]))
        else:
            print('unknown element type!')
            pass
        bl.append(t)
    return bl


def mad2py(fin, fout):
    '''
    read simple/plain MAD8-like (include ELEGANT) lattice input 
    format into python

    fin:  input file name (mad or elegant)
    fout: output pylatt form
    '''
    f1 = open(fin,'r')
    f2 = open(fout,'w')
    f2.write('import pylatt as latt\n\n')
    nline = 0
    while 1:
        a = f1.readline().replace(' ','').upper()
        if '!' in a[1:]:
            comm = string.find(a,'!')
            f2.write('#'+a[comm+1:]+'\n')
            a = a[:comm]+'\n'
        nline += 1
        print('\nLn '+str(nline)+': '+a)
        if not a:
            break
        elif a[0] == '!':
            f2.write('#'+a[1:]+'\n')
        elif a == '\n':
            continue
        else:
            while a[-2] == '&':
                a = a[:-2]+f1.readline().replace(' ','').upper()
                nline += 1
                print('\nLn '+str(nline)+': '+a)
            s = madParse(a[:-1])
            if s:
                f2.write(s+'\n')
    f2.write('ring = latt.cell(RING)')
    f1.close()
    f2.close()


def madParse(a):
    '''
    simple mad8 parse, called by mad2py
    '''
    b = a.split(':')
    name = b[0]
    at = b[1].split(',')
    if at[0] in ['DRIFT','RFCAVITY','MARKER','MARK','RCOL',
                 'MULT','DRIF','RFCA','MALIGN','WATCH']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            else:
                print('  ignored attribute for drif: %s'%rs)
        return('%s = latt.drif(\'%s\', L = %s)'%(name,name,L))
    elif at[0] in ['HKICKER','VKICKER','KICKER','HKICK',
                   'VKICK']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            else:
                print('  ignored attribute for drif: %s'%rs)
        return('%s = latt.kick(\'%s\', L = %s)'%(name,name,L))
    elif at[0] in ['KMAP','UKICKMAP']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            else:
                print('  ignored attribute for kmap: %s'%rs)
        return('%s = latt.kmap(\'%s\', L = %s)'%(name,name,L))
    elif at[0] in ['SCRAPER']:
        L = 0
        apxy = [-1,1,-1,1]
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'POSITION=' in rs:
                aps = rs.split('=')[1]
            elif 'DIRECTION=' in rs:
                drt = rs.split('=')[1]
            else:
                print('  ignored attribute for kmap: %s'%rs)
        drt = int(drt)
        if drt == 0:
            drt = 1
        elif drt == 1:
            drt = 3
        elif drt == 2:
            drt = 0
        elif drt == 3:
            drt = 2
        else:
            raise RuntimeError('unknown aperture direction')
        apxy[drt] = float(aps)
        return('%s = latt.aper(\'%s\', L = %s, aper=[%s,%s,%s,%s])' \
                   %(name,name,L,apxy[0],apxy[1],apxy[2],apxy[3]))
    elif at[0] in ['MONITOR','MONI']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            else:
                print('  ignored attribute for moni: %s'%rs)
        return('%s = latt.moni(\'%s\', L = %s)'%(name,name,L))
    elif at[0] in ['SEXTUPOLE','KSEXT','SEXT']:
        L,K2 = 0,0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'K2=' in rs:
                K2 = rs.split('=')[1]
            else:
                print('  ignored attribute for sext: %s'%rs)
        return('%s = latt.sext(\'%s\', L = %s, K2 = %s)'%(name,name,L,K2))
    elif at[0] in ['QUADRUPOLE','KQUAD','QUAD']:
        L,K1 = 0,0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'K1=' in rs:
                K1 = rs.split('=')[1]
            else:
                print('  ignored attribute for quad: %s'%rs)
        return('%s = latt.quad(\'%s\', L = %s, K1 = %s)'%(name,name,L,K1))
    elif at[0] in ['SBEND','CSBEND','CSBEN','BEND','SBEN']:
        L,K1,K2,E1,E2,ANGLE = 0,0,0,0,0,0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'ANGLE=' in rs:
                ANGLE = rs.split('=')[1]
            elif 'E1=' in rs:
                E1 = rs.split('=')[1]
            elif 'E2=' in rs:
                E2 = rs.split('=')[1]
            elif 'K1=' in rs:
                K1 = rs.split('=')[1]
            elif 'K2=' in rs:
                K2 = rs.split('=')[1]
            else:
                print('  ignored attribute for bend: %s'%rs)
        return('%s = latt.bend(\'%s\',L=%s,angle=%s,e1=%s,e2=%s,K1=%s,K2=%s)' \
                  %(name,name,L,ANGLE,E1,E2,K1,K2))
    elif at[0] == 'RBEND':
        L,K1,K2,E1,E2,ANGLE = 0,0,0,0,0,0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'ANGLE=' in rs:
                ANGLE = rs.split('=')[1]
            elif 'K1=' in rs:
                K1 = rs.split('=')[1]
            elif 'K2=' in rs:
                K2 = rs.split('=')[1]
            else:
                print('  ignored attribute for bend: %s'%rs)
        E1 = ANGLE/2
        E2 = ANGLE/2
        return('%s = latt.bend(\'%s\',L=%s,angle=%s,e1=%s,e2=%s,K1=%s,K2=%s)' \
                  %(name,name,L,ANGLE,E1,E2,K1,K2))
    elif at[0].startswith('LINE'):
        b = a.replace(':LINE=(','=[').replace(')',']')
        return(b)
    else:
        print('unknown type in the mad/elegant file: \'%s\''%at[0])
        return


def chkap(x,xap=[-1.,1.],xpap=[-1.,1],yap=[-1.,1.],ypap=[-1.,1]):
    '''
    check if particles are beyond boundary
    '''
    c1 = np.logical_and(x[0]>xap[0], x[0]<xap[1])
    x = np.compress(c1,x,axis=1)
    c1 = np.logical_and(x[1]>xpap[0], x[1]<xpap[1])
    x = np.compress(c1,x,axis=1)
    c1 = np.logical_and(x[2]>yap[0], x[2]<yap[1])
    x = np.compress(c1,x,axis=1)
    c1 = np.logical_and(x[3]>ypap[0], x[3]<ypap[1])
    x = np.compress(c1,x,axis=1)
    return x


def elempass(ele, x0, nsk=4, nkk=50, nbk=10):
    '''
    element pass
    if element is nonlinear, use kick-drift (sext and kmap)
    otherwise tracking with linear transport matrix
    nsk: number of kick in sext (4 by default)
    nkk: number of kick in kmap (50 by default)
    nbk: number of kick in combined-function dipole
    '''
    x0 = np.mat(np.array(x0)).reshape(6,-1)
    if ele.__class__.__name__ == 'sext':
        K2L = ele.K2*ele.L/nsk
        tm = np.mat(np.eye(6))
        tm[0,1] = ele.L/(nsk+1)
        tm[2,3] = ele.L/(nsk+1)
        x0 = tm*x0
        for m in range(nsk):
            x0[1] -= K2L/2*(np.multiply(x0[0],x0[0])-np.multiply(x0[2],x0[2]))/(1.+x0[5])
            x0[3] += K2L*(np.multiply(x0[0],x0[2]))/(1.+x0[5])
            x0 = tm*x0
        return x0
    elif ele.__class__.__name__ == 'kmap':
        x0 = kdid(ele.kmap1, elekmap2, ele.E, x0, nk=nkk)
        return x0
    elif ele.__class__.__name__ == 'kick':
        tm = np.mat(np.eye(6))
        if ele.L != 0:
            tm[0,1] = ele.L/2.
            tm[2,3] = ele.L/2.
            x0 = tm*x0
            x0[1] += ele.hkick
            x0[3] += ele.vkick
            x0 = tm*x0
        else:
            x0[1] += x0.hkick
            x0[3] += x0.vkick
        return x0
    elif ele.__class__.__name__ == 'bend' and hasattr(ele,'K2'):
        Ld = ele.L/(nbk+1)
        ad = ele.angle/(nbk+1)
        K2L = ele.K2*ele.L/nbk
        B0 = bend('B0',L=Ld,angle=ad,e1=ele.e1,K1=ele.K1)
        BM = bend('BM',L=Ld,angle=ad,K1=ele.K1)
        B1 = bend('B1',L=Ld,angle=ad,e2=ele.e2,K1=ele.K1)
        # --- entrance
        x0 = B0.tm*x0
        # --- middle n-1 kick-drift
        for i in range(nbk-1):
            x0[1] -= K2L/2*(np.multiply(x0[0],x0[0])-np.multiply(x0[2],x0[2]))/(1.+x0[5])
            x0[3] += K2L*(np.multiply(x0[0],x0[2]))/(1.+x0[5])
            x0 = BM.tm*x0
        # --- last kick plus exit
        x0[1] -= K2L/2*(np.multiply(x0[0],x0[0])-np.multiply(x0[2],x0[2]))/(1.+x0[5])
        x0[3] += K2L*(np.multiply(x0[0],x0[2]))/(1.+x0[5])
        x0 = B1.tm*x0
        return x0
    else:
        x0 = ele.tm*x0
        return x0


def readkmap(fn, L, E=3):
    '''
    read kick map from radia output, and scale the kick strength 
    with the given length
    return a dict with scaled length and kick strengthes on the
    same grid as the kick map
    l:      length
    x:      h-axis
    y:      v-axis
    kx:     kick in h-plane (after scaled)
    ky:     kick in v-plane (after scaled)
    '''
    try:
        fid = open(fn,'r')
        a = fid.readlines()
    except:
        raise IOError('I/O error: No such file or directory')
    idlen = float(a[3].strip())
    nx = int(a[5].strip())
    ny = int(a[7].strip())
    thetax = np.empty((ny,nx))
    thetay = np.empty((ny,nx))
    tabx = np.array([float(x) for x in a[10].strip().split() if x != ''])
    taby = np.empty(ny)
    if 'micro-rad' in a[8]:
        unit = 'kick'
    else:
        unit = 'field'
    # --- x-plane
    for m in range(11, 11+ny):
        i = m - 11
        linelist = [float(x) for x in a[m].strip().split() if x != '']
        taby[i] = linelist[0]
        thetax[i] = np.array(linelist[1:])
    # --- y-plane
    for m in range(14+ny, 14+2*ny):
        i = m - (14+ny)
        linelist = [float(x) for x in a[m].strip().split() if x != '']
        thetay[i] = np.array(linelist[1:])
    return {'l':L, 'x':tabx, 'y':taby, 'unit':unit,
            'kx':thetax*L/idlen, 'ky':thetay*L/idlen}


def interp2d(x, y, z, xp, yp):
    '''
    a simple 2d linear interpolation
    x, y:    1-D array
    z:       2-D array must have a size of y row and x col
    xp, yp:  point to interpolate
    
    ----------  x  ---------
    |
    y        z array 
    |
    '''
    if x[-1] < x[0]:
        x = x[::-1]
        z = z[:,::-1]
    if y[-1] < y[0]:
        y = y[::-1]
        z = z[::-1,:]
    xyp = np.array([xp,yp]).reshape(2,-1)
    zp = np.array([])
    for i in range(xyp.shape[1]):
        xi,yi = xyp[0,i],xyp[1,i]
        if np.isnan(xi) or np.isnan(yi):
            zp = np.append(zp,np.nan)
        else:
            if xi>x[-1] or xi<x[0] or yi>y[-1] or yi<y[0]:
                zp = np.append(zp,np.nan)
            else:
                nx2 = np.nonzero(x >= xi)[0][0]
                nx1 = nx2-1
                ny2 = np.nonzero(y >= yi)[0][0]
                ny1 = ny2-1
                x1 = x[nx1]
                x2 = x[nx2]
                y1 = y[ny1]
                y2 = y[ny2]
                f11 = z[ny1, nx1]
                f21 = z[ny1, nx2]
                f12 = z[ny2, nx1]
                f22 = z[ny2, nx2]
                zp = np.append(zp, (f11*(x2-xi)*(y2-yi) + f21*(xi-x1)*(y2-yi) +  \
                                    f12*(x2-xi)*(yi-y1) + f22*(xi-x1)*(yi-y1)) / \
                                   (x2-x1)/(y2-y1))
    return zp


def kdid(kmap1, kmap2, E, x0, nk=50):
    '''
    Kick-Drfit through ID
    usage(kmap1, kmap2, E, X0, nk)

    kamp1:    1st kick map dict (return from readkmap [see readkmap])
    kamp2:    2nd kick map dict (return from readkmap [see readkmap])
    E:        electron nominal energy in GeV
    x0:       vector at entrance, (x, x', y, y', z, delta)
    nk:       number of kicks (100 by default)

    returns
    X:        vector at entrance, (x, x', y, y', z, delta)
    '''
    if kmap1:
        dl = kmap1['l']/(nk+1)
    else:
        dl = kmap2['l']/(nk+1)
    X = np.copy(x0)
    BRho = E*1e9*(1.+X[5])/csp
    X[0] += X[1]*dl
    X[2] += X[3]*dl
    for m in range(nk):
        if kmap1:
            kx = interp2d(kmap1['x'],kmap1['y'],kmap1['kx'],X[0],X[2])
            ky = interp2d(kmap1['x'],kmap1['y'],kmap1['ky'],X[0],X[2])
            X[1] += kx/BRho/nk
            X[3] += ky/BRho/nk
        if kmap2:    
            kx = interp2d(kmap2['x'],kmap2['y'],kmap2['kx'],X[0],X[2])
            ky = interp2d(kmap2['x'],kmap2['y'],kmap2['ky'],X[0],X[2])
            X[1] += kx/BRho/BRho/nk
            X[3] += ky/BRho/BRho/nk
        X[0] += X[1]*dl
        X[2] += X[3]*dl
    return X


def kmap2matrix(kmap1, kmap2, E=3.0, nk=50, dx=1e-5):
    '''
    first order derivative to get a matrix
    E:           electron nominal energy in GeV
    kmap:        kickmap dict
    scale:       kick map scaling factor
    dx:          small coordinate shift to calculate
                 derivative (1e-5 by default)
    '''
    tm = np.eye(6)
    for m in range(6):
        x0 = np.zeros(6)
        x0[m] += dx
        X = kdid(kmap1, kmap2, E, x0, nk=nk)
        tm[:,m] = X/dx
    return tm


def optm(beamline,var,con,wgh,xt=1.e-8,ft=1.e-8,mt=1000,mf=1000):
    '''
    parameters optimization for beamline/cell structure
    beamline: beamline or cell/ring
    var: varible nested list, var[m][0] MUST be an instance
         var=[[q1,'K1'],[q1,'K1']], q1 & q2's K1 are vraible 
    con: constraint list. con[m][0] must be an attribute of beamline
         con=[['betax',0,10.0], ['emitx',5.0]], want betax to be 10m
         at beamline.s[0], and emitx 5.0nm.rad
    wgh: weights for constraints [1,10], 1 for betax, 10 for emitx
    '''
    t0 = time.time()
    if len(con) != len(wgh):
        raise RuntimeError('constraints and weight are not matched')
    val0 = []
    for i,v in enumerate(var):
        m = beamline.bl.index(v[0])
        val0.append(getattr(beamline.bl[m],v[1]))
    val0 = np.array(val0)
    val = opt.fmin(gfun,val0,args=(beamline,var,con,wgh), \
                   xtol=xt,ftol=ft,maxiter=mt,maxfun=mf)
    print('\nOptimized parameters:\n'+21*'-')
    for i,v in enumerate(var):
        m = beamline.bl.index(v[0])
        beamline.bl[m].put(v[1],val[i])
        print('%s: %s, %s = %15.6e'%(beamline.bl[m].name, \
              beamline.bl[m].__class__.__name__, v[1], val[i]))
    print('\nTotally %.2fs is used for optimization.\n'%(time.time()-t0))


def gfun(val,beamline,var,con,wgh):
    '''
    Goal function for optimization
    val: variable value list for optimization
    beamline: beamline or cell/ring
    var: varible nested list, var[m][0] MUST be an instance
         var=[[q1,'K1'],[q1,'K1']], q1 & q2's K1 are vraible 
    con: constraint list. con[m][0] must be an attribute of beamline
         con=[['betax',0,10.0], ['emitx',5.0]], want betax to be 10m
         at beamline.s[0], and emitx 5.0nm.rad
    wgh: weights for constraints [1,10], 1 for betax, 10 for emitx
    '''
    for i,v in enumerate(var):
        m = beamline.bl.index(v[0])
        beamline.bl[m].put(v[1],val[i])
    if beamline.__class__.__name__ == 'beamline':
        beamline.update()
    elif beamline.__class__.__name__ == 'cell':
        fp = [a[0] for a in con]
        rlist = ['alphac','U0','D','Jx','Jy','Je','tau0','taux',
                 'tauy','taue','sige','emitx','sigx']
        clist = ['chx','chx0','chy','chy0']
        ra = any([a in rlist for a in fp])
        ch = any([a in clist for a in fp])
        beamline.update(rad=ra,chrom=ch)
    gf = 0.
    for i,c in enumerate(con):
        if len(c) == 3:
            gf += wgh[i]*(c[2]-getattr(beamline,c[0])[int(c[1])])**2
        else:
            gf += wgh[i]*(c[1]-getattr(beamline,c[0]))**2
    return gf
