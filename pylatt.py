'''
Lattice code in python 2016-12-06

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

__version__ = 2.0
__author__ = 'YONGJUN LI, mpyliyj@gmail.com or mpyliyj@hotmail.com'

import time,copy,string,sys,warnings
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
from scipy.optimize import fmin
from scipy.linalg import logm
from matplotlib.cbook import flatten
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import mvp
from multiprocessing import Process, Queue


np.seterr(all='ignore')

# --- global parameters:
csp = 299792458.0 # speed of light
twopi = 2*np.pi

class drif(object):
    '''
    class: drif - define a drift space with given name and length
    usage: D01 = drif(name='D01',L=1.0,nkick=0,Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    nkick:        number of kicks, reserved for inherited classes
    tag:          tag list for searching
    '''
    def __init__(self,name='D01',L=1.0,nkick=0,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self._nkick = int(nkick)
        self.tag = tag
        self._update()

    def __repr__(self):
        return '%s: %s,L=%g'%(self.name,self.__class__.__name__,self.L)

    def put(self,field,value):
        '''
        update a new value on an instance's attribute
        usage: instance.put(field_name, new_field_value)

        same as use . operator to modify an instance's attribute
        '''
        if hasattr(self,field):
            setattr(self,field,value)
            self._update()
        else:
            raise RuntimeError("%s has no attribute of %s"
                               %(self.name,field))

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self,value):
        try:
            self._L = float(value)
            self._update()
        except:
            raise RuntimeError('L must be float (or convertible)')

    @property
    def Dx(self):
        return self._Dx

    @Dx.setter
    def Dx(self,value):
        try:
            self._Dx = float(value)
            self._update()
        except:
            raise RuntimeError('Dx must be float (or convertible)')

    @property
    def Dy(self):
        return self._Dy

    @Dy.setter
    def Dy(self,value):
        try:
            self._Dy = float(value)
            self._update()
        except:
            raise RuntimeError('Dy must be float (or convertible)')

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self,value):
        try:
            self._tilt = float(value)
            self._update()
        except:
            raise RuntimeError('tilt must be float (or convertible)')

    @property
    def nkick(self):
        return self._nkick

    @nkick.setter
    def nkick(self,value):
        try:
            self._nkick = int(value)
            self._update()
        except:
            raise RuntimeError('nkick must be int (or convertible)')

    @property
    def tm(self):
        return self._tm

    @property
    def tx(self):
        return self._tx

    @property
    def ty(self):
        return self._ty

    def _update(self):
        '''
        update matrices and check magnet length using latest parametes
        '''
        self._chklength()
        self._transmatrix()
        self._twissmatrix()

    def _transmatrix(self):
        '''
        calculate linear transport matrix
        '''
        self._tm = np.eye(6)
        self._tm[0,1] = self.L
        self._tm[2,3] = self.L

    def _twissmatrix(self):
        '''
        from transport matrix to calculate two twiss matrices
        '''
        self._tx = trans2twiss(self.tm[0:2,0:2])
        self._ty = trans2twiss(self.tm[2:4,2:4])

    def _chklength(self):
        '''
        check element length, print a warnning if negative length
        '''
        if self.L < 0.:
            print('warning: %s has a negative length %g'%(self.name,self.L))

    def sympass4(self,x,fast=1):
        '''
        implement 4-th order symplectic pass with initial condition
        x: the initial coordinates in phase space for m particles
        if fast = 1, assume x has (6,-1), otherwise reshape x
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        if self.L != 0:
            x = self.tm.dot(x)
            x[4] += (np.sqrt(1.+np.square(x[1])+np.square(x[3]))-1.)*self.L
        return x


class quad(drif):
    '''
    class: quad - define a quadrupole with given length and K1
    usage: Q01 = quad(name='Q01',L=0.5,K1=0.5)

    Parameter list:
    name:         element name
    L:            length
    K1:           normalized K1 as the MAD convention
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    nkick:        number of kicks
    tag:          tag list for searching
    '''
    def __init__(self,name='Q01',L=0.25,K1=1,nkick=4,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._K1 = float(K1)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self._update()

    def __repr__(self):
        return '%s: %s,L=%g,K1=%g,tilt=%g'%(
            self.name,self.__class__.__name__,self.L,self.K1,self.tilt)

    @property
    def K1(self):
        return self._K1

    @K1.setter
    def K1(self,value):
        try:
            self._K1 = float(value)
            self._update()
        except:
            raise RuntimeError('K1 must be float (or convertible)')

    def _transmatrix(self):
        self._tm = np.eye(6)
        if self.K1 > 0:
            k = np.sqrt(self.K1)
            p = k*self.L
            self._tm[0:2,0:2] = np.array([[np.cos(p),np.sin(p)/k],
                                          [-k*np.sin(p),np.cos(p)]])
            self._tm[2:4,2:4] = np.array([[np.cosh(p),np.sinh(p)/k],
                                          [k*np.sinh(p),np.cosh(p)]])
        elif self.K1 < 0:
            k = np.sqrt(-self.K1)
            p = k*self.L
            self._tm[0:2,0:2] = np.array([[np.cosh(p),np.sinh(p)/k],
                                          [k*np.sinh(p),np.cosh(p)]])
            self._tm[2:4,2:4] = np.array([[np.cos(p),np.sin(p)/k],
                                          [-k*np.sin(p),np.cos(p)]])
        else:
            super(quad,self)._transmatrix()

        if self.tilt != 0.:
            r1 = rotmat(-self.tilt)
            r0 = rotmat(self.tilt)
            self._tm = r1.dot(self.tm).dot(r0)

    def _update(self):
        '''
        update transport (tm) and Twiss (tx,ty) matrices with current
        element parameters, settings for 4th order symplectic pass
        '''
        super(quad,self)._update()
        self._setSympass()

    def _setSympass(self):
        '''
        set symplectic pass
        '''
        if self.K1==0 or self.L==0:
            attrlist = ["_dL","_Ma","_Mb","_K1Lg","_K1Ld"]
            for al in attrlist:
                if hasattr(self,al): delattr(self, al)
            return
        a =  0.675603595979828664
        b = -0.175603595979828664
        g =  1.351207191959657328
        d = -1.702414383919314656
        self._dL = self.L/self.nkick
        self._Ma = np.eye(6)
        self._Ma[0,1] = a*self._dL
        self._Ma[2,3] = self._Ma[0,1]
        self._Mb = np.eye(6)
        self._Mb[0,1] = b*self._dL
        self._Mb[2,3] = self._Mb[0,1]
        self._K1Lg = g*self.K1*self._dL
        self._K1Ld = d*self.K1*self._dL

    def sympass4(self,x,fast=1):
        '''
        implement 4th order symplectic tracking with given initial conditions
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        if self.K1==0 or self.L==0:
            return super(quad,self).sympass4(x)
        else:
            if self.Dx != 0:
                x[0] -= self.Dx
            if self.Dy != 0:
                x[2] -= self.Dy
            if self.tilt != 0:
                x = rotmat(self.tilt).dot(x)
            S = 0.
            for i in range(self.nkick):
                x1p,y1p = x[1],x[3]
                x =  self._Ma.dot(x)
                x[1] -= self._K1Lg*x[0]/(1.+x[5])
                x[3] += self._K1Lg*x[2]/(1.+x[5])
                x =  self._Mb.dot(x)
                x[1] -= self._K1Ld*x[0]/(1.+x[5])
                x[3] += self._K1Ld*x[2]/(1.+x[5])
                x =  self._Mb.dot(x)
                x[1] -= self._K1Lg*x[0]/(1.+x[5])
                x[3] += self._K1Lg*x[2]/(1.+x[5])
                x =  self._Ma.dot(x)
                x2p,y2p = x[1],x[3]
                xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
                # --- average slope at entrance and exit
                S += np.sqrt(1.+np.square(xp)+np.square(yp))*self._dL
            if self.tilt != 0:
                x = rotmat(-self.tilt).dot(x)
            if self.Dy != 0:
                x[2] += self.Dy
            if self.Dx != 0:
                x[0] += self.Dx
            x[4] += S-self.L
            return x


class matr(drif):
    '''
    class matr: define a linear element with its given 6x6 matrix
    usage: M01 = matr(name='M01',tm=np.eye(6),Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    K1:           normalized K1 as the MAD convention
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching
    '''
    def __init__(self,name='MAT01',L=0,tm=np.eye(6),Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._tm = np.array(tm).reshape(6,6)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self._update()

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self,value):
        try:
            self._tm = np.array(value).reshape(6,6)
            self._update()
        except:
            raise RuntimeError('tm must be 6x6 float (or convertible)')

    def _transmatrix(self):
        '''
        if the given matrix is not symplectic, print warning
        '''
        if abs(np.linalg.det(self.tm)-1.) > 1.e-6:
            print('warning: %s\'s linear matrix is not symplectic'%self.name)
        if self.tilt != 0.:
            r1 = rotmat(-self.tilt)
            r0 = rotmat(self.tilt)
            self._tm = r1.dot(self.tm).dot(r0)

    def sympass4(self,x,fast=1):
        '''
        path-length calculation is using the average of slopes at both
        entrance and exit, which might not be so accurate
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        x1p,y1p = x[1],x[3]
        x = np.dot(self.tm,x)
        x2p,y2p = x[1],x[3]
        xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
        x[4] += (np.sqrt(1.+np.square(xp)+np.square(yp))-1.)*self.L
        return x


class moni(drif):
    '''
    class moni: define monitor (BPM) with a default length 0
    usage: BPM01 = moni(name='BPM01',L=0,Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching
    '''
    def __init__(self,name='BPM01',L=0,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self._update()


class rfca(drif):
    '''
    class rfca: define RF cavity with given parameters
    usage: RFC01 = rfca(name='RFC01',voltage=2e6,freq=0.5e9,L=0)

    Parameter list:
    name:         element name
    L:            length
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching
    voltage:      RF cavity voltage in V
    freq:         RF cavity frequency in Hz
    phase:        RF acclerator phase in degree
    '''
    def __init__(self,name='RFC01',L=0,voltage=2e6,
                 freq=0.5e9,phase=0,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self._voltage = float(voltage)
        self._freq = float(freq)
        self._phase = float(phase)
        self.tag = []
        self._update()

    @property
    def voltage(self):
        return self._voltage

    @voltage.setter
    def voltage(self,value):
        try:
            self._voltage = float(value)
            self.update()
        except:
            raise RuntimeError('voltage must be float (or convertible)')

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self,value):
        try:
            self._freq = float(value)
            self.update()
        except:
            raise RuntimeError('freq must be float (or convertible)')

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self,value):
        try:
            self._phase = float(value)
            self.update()
        except:
            raise RuntimeError('phase must be float (or convertible)')


class kick(drif):
    '''
    class kick: define a kick
    usage: K01 = kick(name='K01',L=0,hkick=0,vkick=0,
                      nkick=1,Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    hkick:        horizontal kick, radian
    vkick:        vertical kick, radian
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching
    '''
    def __init__(self,name='K01',L=0,hkick=0,vkick=0,
                 nkick=1,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._hkick = float(hkick)
        self._vkick = float(vkick)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self._update()

    def __repr__(self):
        return '%s: %s, L=%g, hkick=%g, vkick=%g'%(
            self.name,self.__class__.__name__,self.L,self.hkick,self.vkick)

    @property
    def hkick(self):
        return self._hkick

    @hkick.setter
    def hkick(self,value):
        try:
            self._hkick = float(value)
            self._update()
        except:
            raise RuntimeError('hkick must be float (or convertible)')

    @property
    def vkick(self):
        return self._vkick

    @vkick.setter
    def vkick(self,value):
        try:
            self._vkick = float(value)
            self._update()
        except:
            raise RuntimeError('vkick must be float (or convertible)')

    def _update(self):
        super(kick,self)._update()
        if self.L != 0.:
            self._setSympass()
        else:
            self._nkick = 1

    def _setSympass(self):
        '''
        set symplectic pass
        '''
        if self.hkick==0 and self.vkick==0:
            attrlist = ["_dL","_Ma","_Mb","_KhLg","_KhLd","_KvLg","_KvLd"]
            for al in attrlist:
                if hasattr(self,al): delattr(self, al)
            return
        a =  0.675603595979828664
        b = -0.175603595979828664
        g =  1.351207191959657328
        d = -1.702414383919314656
        self._dL = self.L/self.nkick
        self._Ma = np.eye(6)
        self._Ma[0,1] = a*self._dL
        self._Ma[2,3] = self._Ma[0,1]
        self._Mb = np.eye(6)
        self._Mb[0,1] = b*self._dL
        self._Mb[2,3] = self._Mb[0,1]
        self._KhLg = g*self.hkick/self.nkick
        self._KvLg = g*self.vkick/self.nkick
        self._KhLd = d*self.hkick/self.nkick
        self._KvLd = d*self.vkick/self.nkick

    def sympass4(self,x,fast=1):
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        if self.hkick==0 and self.vkick==0:
            return super(kick,self).sympass4(x)
        if self.tilt != 0:
            x = np.dot(rotmat(self.tilt),x)
        if self.L != 0:
            S = 0
            for i in xrange(self.nkick):
                x1p,y1p = x[1],x[3]
                x =  self._Ma.dot(x)
                x[1] += self._KhLg/(1.+x[5])
                x[3] += self._KvLg/(1.+x[5])
                x =  self._Mb.dot(x)
                x[1] += self._KhLd/(1.+x[5])
                x[3] += self._KvLd/(1.+x[5])
                x =  self._Mb.dot(x)
                x[1] += self._KhLg/(1.+x[5])
                x[3] += self._KvLg/(1.+x[5])
                x =  self._Ma.dot(x)
                x2p,y2p = x[1],x[3]
                xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
                S += np.sqrt(1.+np.square(xp)+np.square(yp))*self._dL
            x[4] += S-self.L
        else:
            x[1] += self.hkick/(1+x[5])
            x[3] += self.vkick/(1+x[5])
        if self.tilt != 0:
            x = np.dot(rotmat(-self.tilt),x)
        return x


class aper(drif):
    '''
    class: aper - define a physical aperture with two dimension constraints
    usage: AP01 = aper(name='AP01',L=0,aper=[-0.1,0.1,-0.1,0.1],
                       Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    aper:         rectangle aperture dimensions
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching

    aperture is specfied by 4 parameters in sequence:
    [x_min,x_max,y_min,y_max]
    correctness of aperture configuration will be self-checked
    '''
    def __init__(self,name='APER',L=0,aper=[-1.,1.,-1.,1.],
                 Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._aper = np.array(aper,float)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self._update()

    @property
    def aper(self):
        return self._aper

    @aper.setter
    def aper(self,value):
        try:
            self._aper = np.array(value,float)
            self._update()
        except:
            raise RuntimeError('aper must be 4 floats (or convertible)')

    def _update(self):
        super(aper,self)._update()
        self._selfcheck()

    def _selfcheck(self):
        '''
        check aperture dimension, if OK, no return, otherwise print warning
        '''
        if len(self.aper) != 4:
            print('warning: %s\'s aperture dimension is not 4'%self.name)
        if self.aper[0]>=self.aper[1] or self.aper[2]>=self.aper[3]:
            print('warning: %s\'s aperture format [x_min,x_max,y_min,y_max]'
                  %self.name)

    '''
    def sympass4(self,x0):
        x = super(aper,self).sympass4(x0)
        survived = chkap(x,xap=(self.aper[0],self.aper[1]),
                         yap=(self.aper[2],self.aper[3]))
        survived = np.array(survived)[0]
        for i,si in enumerate(survived):
            if not si:
                x[0,i] = np.nan
                x[2,i] = np.nan
        return x
    '''

class octu(drif):
    '''
    class: octu - define octupole with K3
    usage: OCT01 = octu(name='OCT01',L=0.1,K3=10,nkick=4,
                        Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    K3:           octupole K3 - MAD convention
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    nkick:        number of kicks, reserved for inherited classes
    tag:          tag list for searching
    '''
    def __init__(self,name='OCT01',L=0,K3=0,nkick=4,
                 Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._K3 = float(K3)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self._update()

    def __repr__(self):
        return '%s: %s, L = %g, K3 = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.K3)

    @property
    def K3(self):
        return self._K3

    @K3.setter
    def K3(self,value):
        try:
            self._K3 = float(value)
            self._update()
        except:
            raise RuntimeError('K3 must be float (or convertible)')

    def _update(self):
        '''
        update transport (M) and Twiss (Nx,y) matrices with current
        element parameters, settings for 4th order symplectic pass
        '''
        super(octu,self)._update()
        self._setSympass()

    def _setSympass(self):
        '''
        set symplectic pass
        '''
        if self.K3==0 or self.L==0:
            attrlist = ["_dL","_Ma","_Mb","_K3Lg","_K3Ld"]
            for al in attrlist:
                if hasattr(self,al): delattr(self, al)
            return
        a =  0.675603595979828664
        b = -0.175603595979828664
        g =  1.351207191959657328
        d = -1.702414383919314656
        self._dL = self.L/self.nkick
        self._Ma = np.eye(6)
        self._Ma[0,1] = a*self._dL
        self._Ma[2,3] = self._Ma[0,1]
        self._Mb = np.eye(6)
        self._Mb[0,1] = b*self._dL
        self._Mb[2,3] = self._Mb[0,1]
        self._K3Lg = g*self.K3*self._dL
        self._K3Ld = d*self.K3*self._dL

    def sympass4(self,x,fast=1):
        '''
        implement tracking with 4th order symplectic integrator
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        if self.K3==0 or self.L==0:
            return super(octu,self).sympass4(x)
        else:
            if self.Dx != 0:
                x[0] -= self.Dx
            if self.Dy != 0:
                x[2] -= self.Dy
            if self.tilt != 0:
                x = np.dot(rotmat(self.tilt),x)
            S = 0.
            for i in range(self.nkick):
                x1p,y1p = x[1],x[3]
                x =  np.dot(self._Ma,x)
                x[1] -= self._K3Lg/6*(np.multiply(np.multiply(x[0],x[0]),x[0]) - \
                                      3*np.multiply(x[0],np.multiply(x[2],x[2])))/(1.+x[5])
                x[3] -= self._K3Lg/6*(np.multiply(np.multiply(x[2],x[2]),x[2]) - \
                                     3*np.multiply(x[2],np.multiply(x[0],x[0])))/(1.+x[5])
                x =  np.dot(self._Mb,x)
                x[1] -= self._K3Ld/6*(np.multiply(np.multiply(x[0],x[0]),x[0]) - \
                                     3*np.multiply(x[0],np.multiply(x[2],x[2])))/(1.+x[5])
                x[3] -= self._K3Ld/6*(np.multiply(np.multiply(x[2],x[2]),x[2]) - \
                                     3*np.multiply(x[2],np.multiply(x[0],x[0])))/(1.+x[5])
                x =  np.dot(self._Mb,x)
                x[1] -= self._K3Lg/6*(np.multiply(np.multiply(x[0],x[0]),x[0]) - \
                                     3*np.multiply(x[0],np.multiply(x[2],x[2])))/(1.+x[5])
                x[3] -= self._K3Lg/6*(np.multiply(np.multiply(x[2],x[2]),x[2]) - \
                                     3*np.multiply(x[2],np.multiply(x[0],x[0])))/(1.+x[5])
                x =  np.dot(self._Ma,x)
                x2p,y2p = x[1],x[3]
                xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
                S += np.sqrt(1.+np.square(xp)+np.square(yp))*self._dL
            if self.tilt != 0:
                x = np.dot(rotmat(-self.tilt),x)
            if self.Dy != 0:
                x[2] += self.Dy
            if self.Dx != 0:
                x[0] += self.Dx
            x[4] += S-self.L
            return x
        #    x[1] -= self.K3/6*(np.multiply(np.multiply(x[0],x[0]),x[0]) - \
        #                         3*np.multiply(x[0],np.multiply(x[2],x[2])))/(1.+x[5])
        #    x[3] -= self.K3/6*(np.multiply(np.multiply(x[2],x[2]),x[2]) - \
        #                         3*np.multiply(x[2],np.multiply(x[0],x[0])))/(1.+x[5])
        #    return x


class sext(drif):
    '''
    class: sext - define setupole with length and K2
    usage: SEXT01 = sext(name='SEXT01',L=0.25,K2=1,
                         nkick=4,Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    K2:           octupole K2 - MAD convention
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    nkick:        number of kicks, reserved for inherited classes
    tag:          tag list for searching
    '''
    def __init__(self,name='SEXT01',L=0.25,K2=1,nkick=4,
                 Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._K2 = float(K2)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = []
        self._update()

    def __repr__(self):
        return '%s: %s, L = %g, K2 = %15.8f'%(
            self.name,self.__class__.__name__,self.L,self.K2)

    @property
    def K2(self):
        return self._K2

    @K2.setter
    def K2(self,value):
        try:
            self._K2 = float(value)
            self._update()
        except:
            raise RuntimeError('K2 must be float (or convertible)')

    def _update(self):
        '''
        update transport (M) and Twiss (Nx,y) matrices with current
        element parameters, settings for 4th order symplectic pass
        '''
        super(sext,self)._update()
        self._setSympass()

    def _setSympass(self):
        '''
        set symplectic pass
        '''
        if self.K2==0 or self.L==0:
            attrlist = ["_dL","_Ma","_Mb","_K2Lg","_K2Ld"]
            for al in attrlist:
                if hasattr(self,al): delattr(self, al)
            return
        a =  0.675603595979828664
        b = -0.175603595979828664
        g =  1.351207191959657328
        d = -1.702414383919314656
        self._dL = self.L/self.nkick
        self._Ma = np.eye(6)
        self._Ma[0,1] = a*self._dL
        self._Ma[2,3] = self._Ma[0,1]
        self._Mb = np.eye(6)
        self._Mb[0,1] = b*self._dL
        self._Mb[2,3] = self._Mb[0,1]
        self._K2Lg = g*self.K2*self._dL
        self._K2Ld = d*self.K2*self._dL

    def sympass4(self,x,fast=1):
        '''
        implement tracking with 4th order symplectic integrator
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        if self.K2==0 or self.L==0:
            return super(sext,self).sympass4(x)
        else:
            if self.Dx != 0:
                x[0] -= self.Dx
            if self.Dy != 0:
                x[2] -= self.Dy
            if self.tilt != 0:
                x = np.dot(rotmat(self.tilt),x)
            S = 0.
            for i in range(self.nkick):
                x1p,y1p = x[1],x[3]
                x =  np.dot(self._Ma,x)
                x[1] -= self._K2Lg/2*(np.multiply(x[0],x[0]) - \
                                     np.multiply(x[2],x[2]))/(1.+x[5])
                x[3] += self._K2Lg*(np.multiply(x[0],x[2]))/(1.+x[5])
                x =  np.dot(self._Mb,x)
                x[1] -= self._K2Ld/2*(np.multiply(x[0],x[0]) - \
                                     np.multiply(x[2],x[2]))/(1.+x[5])
                x[3] += self._K2Ld*(np.multiply(x[0],x[2]))/(1.+x[5])
                x =  np.dot(self._Mb,x)
                x[1] -= self._K2Lg/2*(np.multiply(x[0],x[0]) - \
                                     np.multiply(x[2],x[2]))/(1.+x[5])
                x[3] += self._K2Lg*(np.multiply(x[0],x[2]))/(1.+x[5])
                x =  np.dot(self._Ma,x)
                x2p,y2p = x[1],x[3]
                xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
                S += np.sqrt(1.+np.square(xp)+np.square(yp))*self._dL
            if self.tilt != 0:
                x = np.dot(rotmat(-self.tilt),x)
            if self.Dy != 0:
                x[2] += self.Dy
            if self.Dx != 0:
                x[0] += self.Dx
            x[4] += S-self.L
            return x


class kmap(drif):
    '''
    class: kmap - define a RADIA kickmap from an external text file
    usage: ID01 = kmap(name='ID01',L=1.2,kmap2fn='cwd/ivu29_kmap.txt',E=3)

    kmap1fn: 1st order kickmap (strip line) file name with directory info
    kmap2fn: 2nd order kickmap (magnet york) file name with directory info
    E: beam energy in GeV

    1. kickmap file itself includes a length. This definition will scale
    with the integral to the actual device length, i.e. L.

    2. kickmap includes the magnetic field [Tm/T2m2] or kicks [micro-rad]
    normalized with a certain beam energy (3Gev by default). The kick unit
    is re-normalized to field unit.
    '''
    def __init__(self,name='ID01', L=0,
                 kmap1fn=None, kmap2fn=None,
                 E=3,nkick=20,Dx=0,Dy=0,tilt=0,Bw=1.8):
        self.name = str(name)
        self._L = float(L)
        self._kmap1fn = kmap1fn
        self._kmap2fn = kmap2fn
        self._E = float(E)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self._Bw = Bw
        BRho = self.E*1e9/csp
        if self.kmap1fn:
            self._kmap1 = self._readkmap(self.kmap1fn)
            if self._kmap1['unit'] == 'kick':
                self._kmap1['kx'] = self._kmap1['kx']*1e-6*BRho
                self._kmap1['ky'] = self._kmap1['ky']*1e-6*BRho
                self._kmap1['unit'] = 'field'
        else:
            self._kmap1 = None
        if self.kmap2fn:
            self._kmap2 = self._readkmap(self.kmap2fn)
            if self._kmap2['unit'] == 'kick':
                self._kmap2['kx'] = self._kmap2['kx']*1e-6*BRho*BRho
                self._kmap2['ky'] = self._kmap2['ky']*1e-6*BRho*BRho
                self._kmap2['unit'] = 'field'
        else:
            self._kmap2 = None
        self._update()

    def __repr__(self):
        return "%s: %s, L = %g, E =%8.4f, kmap1fn = '%s', kmap2fn = '%s'"%(
            self.name,self.__class__.__name__,self.L,self.E,
            self.kmap1fn,self.kmap2fn)

    @property
    def kmap1fn(self):
        return self._kmap1fn

    @kmap1fn.setter
    def kmap1fn(self,value):
        try:
            self._kmap1fn = str(value)
            self._update()
        except:
            raise RuntimeError('kmap1fn must be string (or convertible)')

    @property
    def kmap2fn(self):
        return self._kmap2fn

    @kmap2fn.setter
    def kmap2fn(self,value):
        try:
            self._kmap2fn = str(value)
            self._update()
        except:
            raise RuntimeError('kmap2fn must be string (or convertible)')

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self,value):
        try:
            self._E = float(value)
            self._update()
        except:
            raise RuntimeError('E must be float (or convertible)')
    @property
    def Bw(self):
        return self._Bw

    @Bw.setter
    def Bw(self,value):
        try:
            self._Bw = float(value)
            self._update()
        except:
            raise RuntimeError('Bw must be float (or convertible)')

    def _transmatrix(self,dx=1e-4):
        '''
        calculate transport matrix from kick map files
        first, read kick maps from given kick map files
        then, if kick map unit is [micro-rad],
        it will be un-normailized by beam energy, and reset its unit as [field]
        '''
        tm = np.eye(6)
        for m in range(4):
            x0 = np.zeros(6)
            x0[m] += dx
            x = self.sympass4(x0)
            tm[:,m] = x/dx
        tm[4,:4] = 0
        self._tm = tm
        if self.tilt != 0.:
            r1 = rotmat(-self.tilt)
            r0 = rotmat(self.tilt)
            self._tm = r1.dot(self.tm).dot(r0)

    def _readkmap(self,fn):
        '''
        read kick map from "radia" output, and scale the kick strength
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
        tabx = np.array([float(x) for x in a[10].strip().split() if x!=''])
        taby = np.empty(ny)
        if 'micro-rad' in a[8]:
            unit = 'kick'
        else:
            unit = 'field'
        # --- x-plane
        for m in range(11, 11+ny):
            i = m - 11
            linelist = [float(x) for x in a[m].strip().split() if x!='']
            taby[i] = linelist[0]
            thetax[i] = np.array(linelist[1:])
        # --- y-plane
        for m in range(14+ny, 14+2*ny):
            i = m - (14+ny)
            linelist = [float(x) for x in a[m].strip().split() if x!='']
            thetay[i] = np.array(linelist[1:])
        return {'x':tabx, 'y':taby, 'unit':unit,
                'kx':thetax*self.L/idlen, 'ky':thetay*self.L/idlen}

    def sympass4(self,x,fast=1):
        '''
        Kick-Drfit through ID
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        dl = self.L/(self.nkick+1)
        BRho = self.E*1e9*(1.+x[5])/csp
        if self.Dx != 0:
            x[0] -= self.Dx
        if self.Dy != 0:
            x[2] -= self.Dy
        if self.tilt != 0:
            x = rotmat(self.tilt).dot(x)
        S = 0.
        x[0] += x[1]*dl
        x[2] += x[3]*dl
        S += np.sqrt(1.+np.square(x[1])+np.square(x[3]))*dl
        for m in xrange(self.nkick):
            if self._kmap1:
                kx = interp2d(self._kmap1['x'],self._kmap1['y'],
                              self._kmap1['kx'],x[0],x[2])
                ky = interp2d(self._kmap1['x'],self._kmap1['y'],
                              self._kmap1['ky'],x[0],x[2])
                x[1] += kx/BRho/self.nkick
                x[3] += ky/BRho/self.nkick
            if self._kmap2:
                kx = interp2d(self._kmap2['x'],self._kmap2['y'],
                              self._kmap2['kx'],x[0],x[2])

                ky = interp2d(self._kmap2['x'],self._kmap2['y'],
                              self._kmap2['ky'],x[0],x[2])
                x[1] += kx/BRho/BRho/self.nkick
                x[3] += ky/BRho/BRho/self.nkick
            x[0] += x[1]*dl
            x[2] += x[3]*dl
            S += np.sqrt(1.+np.square(x[1])+np.square(x[3]))*dl
        x[4] += S-self.L
        if self.tilt != 0:
            x = rotmat(-self.tilt).dot(x)
        if self.Dy != 0:
            x[2] += self.Dy
        if self.Dx != 0:
            x[0] += self.Dx
        return x


class bend(drif):
    '''
    class: bend - define a bend (dipole) with given parmeters
    usage: B01 = bend(name='B01',L=1,angle=0.2,e1=0.1,e2=0.1,K1=0,
                      K2=0,nkick=10,hgap=0,fint=0.5,Dx=0,Dy=0,tilt=0,tag=[])

    Parameter list:
    name:         element name
    L:            length
    angle:        bending angle
    e1, e2:       fringe angles at entrance/exit
    K1, K2:       integrated quad, sext component
    nkick:        number of kicks for symplectic tracking
    hgap:         half gap between two poles
    fint:         fringe field integral
    Dx, Dy, tilt: misalignment in meter, radian
    tm:           transport matrix 6x6
    tx,ty:        twiss matrics 3x3 for x and y plane
    tag:          tag list for searching

    notice: it can have quadrupole and sextupole gradients integrated
    '''
    def __init__(self,name='B01',L=1,angle=1e-9,\
                 e1=0,e2=0,K1=0,K2=0,nkick=10,\
                 hgap=0,fint=0.5,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._angle = float(angle)
        self._e1 = float(e1)
        self._e2 = float(e2)
        self._K1 = float(K1)
        self._K2 = float(K2)
        self._nkick = int(nkick)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self._hgap = float(hgap)
        self._fint = float(fint)
        self.tag = []
        self._update()

    def __repr__(self):
        return '%s: %s, L=%g, angle=%g, e1=%g, e2=%g'%(
            self.name,self.__class__.__name__,self.L,self.angle,self.e1,self.e2)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self,value):
        try:
            self._angle = float(value)
            self._update()
        except:
            raise RuntimeError('angle must be non-zero float (or convertible)')

    @property
    def e1(self):
        return self._e1

    @e1.setter
    def e1(self,value):
        try:
            self._e1 = float(value)
            self._update()
        except:
            raise RuntimeError('e1 must be float (or convertible)')

    @property
    def e2(self):
        return self._e2

    @e2.setter
    def e2(self,value):
        try:
            self._e2 = float(value)
            self._update()
        except:
            raise RuntimeError('e2 must be float (or convertible)')

    @property
    def K1(self):
        return self._K1

    @K1.setter
    def K1(self,value):
        try:
            self._K1 = float(value)
            self._update()
        except:
            raise RuntimeError('K1 must be float (or convertible)')

    @property
    def K2(self):
        return self._K2

    @K2.setter
    def K2(self,value):
        try:
            self._K2 = float(value)
            self._update()
        except:
            raise RuntimeError('K2 must be float (or convertible)')

    @property
    def hgap(self):
        return self._hgap

    @hgap.setter
    def hgap(self,value):
        try:
            self._hgap = float(value)
            self._update()
        except:
            raise RuntimeError('hgap must be float (or convertible)')

    @property
    def fint(self):
        return self._fint

    @fint.setter
    def fint(self,value):
        try:
            self._fint = float(value)
            self._update()
        except:
            raise RuntimeError('fint must be float (or convertible)')

    @property
    def R(self):
        return self._R

    def _transmatrix(self):
        kx = self.K1+1/(self._R*self._R)
        self._tm = np.eye(6)
        if kx > 0:
            k = np.sqrt(kx)
            p = k*self.L
            self._tm[0:2,0:2] = np.array([[np.cos(p),np.sin(p)/k],
                                          [-k*np.sin(p),np.cos(p)]])
            self._tm[0:2,5:6] = np.array([[(1-np.cos(p))/(self._R*kx)],
                                          [np.sin(p)/(self._R*k)]])
            self._tm[4,0] = self.tm[1,5]
            self._tm[4,1] = self.tm[0,5]
            self._tm[4,5] = (k*self.L-np.sin(p))/self._R/self._R/k/kx
        elif kx < 0:
            k = np.sqrt(-kx)
            p = k*self.L
            self._tm[0:2,0:2] = np.array([[np.cosh(p),np.sinh(p)/k],
                                          [k*np.sinh(p),np.cosh(p)]])
            self._tm[0:2,5:6] = np.array([[(np.cosh(p)-1)/(-self._R*kx)],
                                          [np.sinh(p)/(self._R*k)]])
            self._tm[4,0] = self.tm[1,5]
            self._tm[4,1] = self.tm[0,5]
            self._tm[4,5] = (k*self.L-np.sinh(p))/self._R/self._R/k/kx
        else:
            self._tm[0,1] = self.L
        if self.K1 > 0:
            k = np.sqrt(self.K1)
            p = k*self.L
            self._tm[2:4,2:4] = np.array([[np.cosh(p),np.sinh(p)/k],
                                          [k*np.sinh(p),np.cosh(p)]])
        elif self.K1 < 0:
            k = np.sqrt(-self.K1)
            p = k*self.L
            self._tm[2:4,2:4] = np.array([[np.cos(p),np.sin(p)/k],
                                          [-k*np.sin(p),np.cos(p)]])
        else:
           self._tm[2,3] = self.L
        #entrance
        if self.hgap!=0 and self.fint!=0:
            vf  = -1/self._R*self.hgap*self.fint * \
                 (1+np.sin(self.e1)**2)/np.cos(self.e1)*2
        else:
            vf = 0
        if self.e1!=0 or vf!=0:
            m1 = np.eye(6)
            m1[1,0] = np.tan(self.e1)/self._R
            m1[3,2] = -np.tan(self.e1+vf)/self._R
            self._m1 = m1
            self._tm = self.tm.dot(m1)
        #exit
        if self.hgap!=0 and self.fint!=0:
            vf = -1/self._R*self.hgap*self.fint * \
                (1+np.sin(self.e2)**2)/np.cos(self.e2)*2
        else:
            vf = 0
        if self.e1!=0 or vf!=0:
            m2 = np.eye(6)
            m2[1,0] = np.tan(self.e2)/self._R
            m2[3,2] = -np.tan(self.e2+vf)/self._R
            self._m2 = m2
            self._tm = m2.dot(self.tm)
        if self.tilt != 0.:
            r1 = rotmat(-self.tilt)
            r0 = rotmat(self.tilt)
            self._tm = r1.dot(self.tm).dot(r0)

    def _update(self):
        '''
        update transport (M) and Twiss (Nx,y) matrices with current
        element parameters, settings for 4th order symplectic pass
        '''
        try:
            self._R = self.L/self.angle
        except:
            raise RuntimeError("%s: bending angle is zero"%self.name)
        super(bend,self)._update()
        self._setSympass()

    def _setSympass(self):
        '''
        set symplectic pass
        '''
        a =  0.675603595979828664
        b = -0.175603595979828664
        g =  1.351207191959657328
        d = -1.702414383919314656
        self._dL = self.L/self.nkick
        self._Ma = np.eye(6)
        self._Ma[0,1] = a*self._dL
        self._Ma[2,3] = self._Ma[0,1]
        self._Mb = np.eye(6)
        self._Mb[0,1] = b*self._dL
        self._Mb[2,3] = self._Mb[0,1]
        self._Lg = g*self._dL
        self._Ld = d*self._dL
        self._K1Lg = g*self.K1*self._dL
        self._K1Ld = d*self.K1*self._dL
        self._K2Lg = g*self.K2*self._dL
        self._K2Ld = d*self.K2*self._dL

    def sympass4(self,x,fast=1):
        '''
        implement tracking with 4yh order symplectic integrator
        '''
        if not fast:
            x = np.array(x,dtype=float).reshape(6,-1)
        S = 0.
        if self.Dx != 0:
            x[0] -= self.Dx
        if self.Dy != 0:
            x[2] -= self.Dy
        if self.tilt != 0:
            x = rotmat(self.tilt)*x
        if hasattr(self,'_m1'):
            x = self._m1.dot(x)
        for i in range(self.nkick):
            x1p,y1p = x[1],x[3]
            x1 = x[0]
            x =  self._Ma.dot(x)
            x[1] -= self._K2Lg/2*(np.multiply(x[0],x[0])-np.multiply(x[2],x[2]))/(1+x[5]) \
                   +self._K1Lg*x[0]/(1+x[5])-self._Lg*x[5]/self._R+self._Lg*x[0]/self._R**2
            x[3] += self._K2Lg*(np.multiply(x[0],x[2]))/(1+x[5])+self._K1Lg*x[2]/(1+x[5])
            x =  self._Mb.dot(x)
            x[1] -= self._K2Ld/2*(np.multiply(x[0],x[0])-np.multiply(x[2],x[2]))/(1+x[5]) \
                   +self._K1Ld*x[0]/(1+x[5])-self._Ld*x[5]/self._R+self._Ld*x[0]/self._R**2
            x[3] += self._K2Ld*(np.multiply(x[0],x[2]))/(1+x[5])+self._K1Ld*x[2]/(1+x[5])
            x =  self._Mb.dot(x)
            x[1] -= self._K2Lg/2*(np.multiply(x[0],x[0])-np.multiply(x[2],x[2]))/(1+x[5]) \
                   +self._K1Lg*x[0]/(1+x[5])-self._Lg*x[5]/self._R+self._Lg*x[0]/self._R**2
            x[3] += self._K2Lg*(np.multiply(x[0],x[2]))/(1+x[5])+self._K1Lg*x[2]/(1+x[5])
            x =  self._Ma.dot(x)
            x2p,y2p = x[1],x[3]
            x2 = x[0]
            xp,yp = (x1p+x2p)/2,(y1p+y2p)/2
            xav = (x1+x2)/2
            S += np.multiply(np.sqrt(1.+np.square(xp)+np.square(yp)),
                             (1+xav/self._R))*self._dL
        if hasattr(self,'_m2'):
            x = self._m2.dot(x)
        if self.tilt != 0:
            x = rotmat(-self.tilt).dot(x)
        if self.Dy != 0:
            x[2] += self.Dy
        if self.Dx != 0:
            x[0] += self.Dx
        x[4] += S-self.L
        return x


class wigg(drif):
    '''
    class: wigg - define a simple model for wiggler with given length and field
    usage: WG01 = wigg(name='WG01',L=1,Bw=1,E=3,Dx=0,Dy=0,tilt=0,tag=[])

    parameter list:
    name:         element name
    L:            length
    Dx, Dy, tilt: misalignment in meter, radian
    Bw:           average wiggler field strength within one half period
    E:            beam energy in GeV used for normalize field
    tag:          tag list used for searching

    Notice:
    this model basically is only used for radiation calculation
    '''
    def __init__(self,name='WG01',L=1,Bw=1,E=3,Dx=0,Dy=0,tilt=0,tag=[]):
        self.name = str(name)
        self._L = float(L)
        self._Bw = float(Bw)
        self._E = float(E)
        self._Dx = float(Dx)
        self._Dy = float(Dy)
        self._tilt = float(tilt)
        self.tag = tag
        self.update()

    def __repr__(self):
        return '%s: %s, L=%g, Bw=%15.8f'%(
            self.name,self.__class__.__name__,self.L,self.Bw)

    @property
    def Bw(self):
        return self._Bw

    @Bw.setter
    def Bw(self,value):
        try:
            self._Bw = float(value)
            self._update()
        except:
            raise RuntimeError('Bw must be float (or convertible)')

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self,value):
        try:
            self._E = float(value)
            self._update()
        except:
            raise RuntimeError('E must be float (or convertible)')

    def _transmatrix(self):
        rhoinv = self.Bw/(self.E*1e9)*csp
        k = abs(rhoinv)/np.sqrt(2.)
        p = k*self.L
        self._tm = np.eye(6)
        self._tm[0,1] = self.L
        self._tm[2:4,2:4] = [[np.cos(p),np.sin(p)/k],
                             [-k*np.sin(p),np.cos(p)]]

        if self.tilt != 0.:
            r1 = rotmat(-self.tilt)
            r0 = rotmat(self.tilt)
            self._tm = r1.dot(self.tm).dot(r0)


class beamline(object):
    '''
    class: beamline - define a beamline with a squence of magnets
    usage: beamline(bl,twx0,twy0,dx0,N=1,E=3)

    twx0, twy0, dx: Twiss paramters at the starting point
                    3x1 matrix
    '''
    def __init__(self,bl,betax0=10,alfax0=0,betay0=10,alfay0=0,
                 etax0=0,etaxp0=0,etay0=0,etayp0=0,
                 emitx=1000,emity=1000,sige=1e-2,E=3):
        self._bl = [ele for ele in flatten(bl)]
        self._E = float(E)
        self._betax0,self._alfax0,self._betay0,self._alfay0 = \
                    betax0,alfax0,betay0,alfay0
        self._etax0,self._etaxp0,self._etay0,self._etayp0 = \
                    etax0,etaxp0,etay0,etayp0
        self._emitx = float(emitx)
        self._emity = float(emity)
        self._sige = float(sige)
        self._update()

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self,value):
        try:
            self._E = float(value)
            self._update()
        except:
            raise RuntimeError('E must be float (or convertible)')

    @property
    def bl(self):
        return self._bl

    @bl.setter
    def bl(self,value):
        try:
            self._bl = list(value)
            self._update()
        except:
            raise RuntimeError('bl must be a list of magnet instances')

    @property
    def betax0(self):
        return self._betax0

    @betax0.setter
    def betax0(self,value):
        try:
            self._betax0 = float(value)
            self._update()
        except:
            raise RuntimeError('betax0 must be float (or convertible)')

    @property
    def alfax0(self):
        return self._alfax0

    @alfax0.setter
    def alfax0(self,value):
        try:
            self._alfax0 = float(value)
            self._update()
        except:
            raise RuntimeError('alfax0 must be float (or convertible)')

    @property
    def betay0(self):
        return self._betay0

    @betay0.setter
    def betay0(self,value):
        try:
            self._betay0 = float(value)
            self._update()
        except:
            raise RuntimeError('betay0 must be float (or convertible)')

    @property
    def alfay0(self):
        return self._alfay0

    @alfay0.setter
    def alfay0(self,value):
        try:
            self._alfay0 = float(value)
            self._update()
        except:
            raise RuntimeError('alfay0 must be float (or convertible)')

    @property
    def etax0(self):
        return self._etax0

    @etax0.setter
    def etax0(self,value):
        try:
            self._etax0 = float(value)
            self._update()
        except:
            raise RuntimeError('etax0 must be float (or convertible)')

    @property
    def etaxp0(self):
        return self._etaxp0

    @etaxp0.setter
    def etaxp0(self,value):
        try:
            self._etaxp0 = float(value)
            self._update()
        except:
            raise RuntimeError('etaxp0 must be float (or convertible)')

    @property
    def etay0(self):
        return self._etay0

    @etay0.setter
    def etay0(self,value):
        try:
            self._etay0 = float(value)
            self._update()
        except:
            raise RuntimeError('etay0 must be float (or convertible)')

    @property
    def etayp0(self):
        return self._etayp0

    @etayp0.setter
    def etayp0(self,value):
        try:
            self._etayp0 = float(value)
            self._update()
        except:
            raise RuntimeError('etayp0 must be float (or convertible)')

    @property
    def emitx(self):
        return self._emitx

    @emitx.setter
    def emitx(self,value):
        try:
            self._emitx = float(value)
            self._update()
        except:
            raise RuntimeError('emitx must be float (or convertible)')

    @property
    def emity(self):
        return self._emity

    @emity.setter
    def emity(self,value):
        try:
            self._emity = float(value)
            self._update()
        except:
            raise RuntimeError('emity must be float (or convertible)')

    @property
    def sige(self):
        return self._sige

    @sige.setter
    def sige(self,value):
        try:
            self._sige = float(value)
            self._update()
        except:
            raise RuntimeError('sige must be float (or convertible)')

    @property
    def s(self):
        return self._s

    @property
    def betax(self):
        return self._betax

    @property
    def betay(self):
        return self._betay

    @property
    def alfax(self):
        return self._alfax

    @property
    def alfay(self):
        return self._alfay

    @property
    def mux(self):
        return self._mux

    @property
    def muy(self):
        return self._muy

    @property
    def etax(self):
        return self._etax

    @property
    def etaxp(self):
        return self._etaxp

    @property
    def etay(self):
        return self._etay

    @property
    def etayp(self):
        return self._etayp

    @property
    def sigx(self):
        return self._sigx

    @property
    def sigy(self):
        return self._sigy

    @property
    def nux(self):
        return self._nux

    @property
    def nuy(self):
        return self._nuy

    @property
    def L(self):
        return self._L

    def _update(self):
        self._bl = self.bl
        self._spos()
        self._twiss()
        self._disp()
        self._sigx = [np.sqrt(self.betax[m]*self.emitx*1e-9)
                      +abs(self.etax[m])*self.sige for m in range(len(self.s))]
        self._sigy = [np.sqrt(self.betay[m]*self.emity*1e-9)
                      for m in range(len(self.s))]

    def __repr__(self):
        s = ''
        s += '\n'+'-'*11*11+'\n'
        s += 11*'%11s'%('s','betax','alfax','mux','etax','etaxp','betay','alfay','muy','etay','etayp')+'\n'
        s += '-'*11*11+'\n'
        s += (11*'%11.3e'+'\n')%(self.s[0],self.betax[0],self.alfax[0],self.mux[0], \
                                 self.etax[0],self.etaxp[0],self.betay[0],\
                                 self.alfay[0],self.muy[0],self.etay[0],self.etayp[0])
        s += (11*'%11.3e'+'\n')%(self.s[-1],self.betax[-1],self.alfax[-1],self.mux[-1], \
                                 self.etax[-1],self.etaxp[-1],self.betay[-1],\
                                 self.alfay[-1],self.muy[-1],self.etay[0],self.etayp[-1])+'\n'
        s += 'Tune: nux = %11.3f, nuy = %11.3f\n'%(self.nux,self.nuy)+'\n'
        return s

    def _spos(self):
        '''
        get s coordinates for each element
        '''
        s = [0]
        for elem in self.bl:
            s.append(s[-1]+elem.L)
        self._s = np.array(s)
        self._L = s[-1]

    def _twiss(self):
        '''
        get twiss parameters along s. twx/y are matrix format,
        betax/y, alfax/y are 1D array, duplicate data are kept for convinence
        '''
        self._mux,self._muy = [0.],[0.]
        self._twx = np.array([self.betax0,self.alfax0
                              ,(1+self.alfax0*self.alfax0)/self.betax0],
                             float).reshape(3,1)
        self._twy = np.array([self.betay0,self.alfay0,
                              (1+self.alfay0*self.alfay0)/self.betay0],
                             float).reshape(3,1)
        self._dxy = np.array([self.etax0,self.etaxp0,self.etay0,self.etayp0,1],
                             float).reshape(-1,1)
        for elem in self.bl:
            mx = elem.tm[0:2,0:2]
            my = elem.tm[2:4,2:4]
            if elem.L < 0:
                neglen = True
            else:
                neglen = False
            self._mux = np.append(self.mux,self.mux[-1]+phasetrans(mx,self._twx[:,-1],neglen=neglen))
            self._muy = np.append(self.muy,self.muy[-1]+phasetrans(my,self._twy[:,-1],neglen=neglen))
            self._twx = np.append(self._twx,twisstrans(elem.tx,self._twx[:,-1]).reshape(3,1),axis=1)
            self._twy = np.append(self._twy,twisstrans(elem.ty,self._twy[:,-1]).reshape(3,1),axis=1)
        self._betax = self._twx[0]
        self._betay = self._twy[0]
        self._alfax = self._twx[1]
        self._alfay = self._twy[1]
        self._gamax = self._twx[2]
        self._gamay = self._twy[2]
        self._nux = self.mux[-1]
        self._nuy = self.muy[-1]

    def _disp(self):
        '''
        get dispersion along s
        '''
        for elem in self.bl:
            m = np.take(np.take(elem.tm,[0,1,2,3,5],axis=0),
                        [0,1,2,3,5],axis=1)
            self._dxy = np.append(self._dxy,m.dot(self._dxy[:,-1]).reshape(-1,1),axis=1)
        self._etax =  self._dxy[0]
        self._etaxp = self._dxy[1]
        self._etay =  self._dxy[2]
        self._etayp = self._dxy[3]


    def getElements(self,types=['aper','bend','drif','kick','kmap',
                                'matr','moni','octu','quad','rfca',
                                'sext','wigg'],
                    prefix='',suffix='',include='',exclude='',
                    unique=False):
        '''
        get all elements in the given types and with element name
        * start with 'prefix'
        * ends with 'suffix'
        * include 'include'
        * exclude 'exclude'
        if lors == 'list' return a list, otherwise return a set
        used for optimization
        '''
        ele = []
        for el in self.bl:
            cond = el.__class__.__name__ in types  \
                   and el.name.startswith(prefix) \
                   and el.name.endswith(suffix) \
                   and include in el.name
            if len(exclude):
                cond = cond and exclude not in el.name
            if cond:
                ele.append(el)
        if unique:
            return list(set(ele))
        else:
            return ele

    def getSIndex(self,s0):
        '''
        get the s index where s0 is in-between
        '''
        s0 = float(s0)
        if s0 <= 0:
            return 0
        elif s0 >= self.L:
            return -1
        else:
            m = np.nonzero(self.s>=s0)[0][0]
            return m-1,m

    def getIndex(self,types,prefix='',suffix='',include='',
                 exclude=''):
        '''
        get all element index with the given types and
        starts/ends with the specified string
        types:    str/list-like, element types
        prefix:   str, which is the string element name starts with
        suffix:   str, which is the string element name ends with
        exitport: bool, entrance or exit, if True exit, otherwise entrance
        '''
        idx = []
        for i,el in enumerate(self.bl):
            cond = el.__class__.__name__ in types  \
                   and el.name.startswith(prefix) \
                   and el.name.endswith(suffix) \
                   and include in el.name
            if len(exclude):
                cond = cond and exclude not in el.name
            if cond:
                idx.append(i)
        return idx

    def pltmag(self,unit=1.0,surflvl=0,alfa=1):
        '''
        plot magnets along s
        '''
        sh = 0.75*unit #sext
        qh = 1.0*unit  #quad
        bh = 0.5*unit  #bend
        kh = 0.25*unit #kmap - insertion device, solenoid
        mc = 0.2*unit  #corr - kick
        mh = 0.15*unit #moni - bpm
        mrf = 0.2*unit #RF cavity
        s = [0.]
        ax = plt.gca()
        for e in self.bl:
            if e.__class__.__name__ in ['aper','drif','matr']:
                pass
            elif e.__class__.__name__ == 'quad':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-qh),e.L,2*qh,fc='y',ec='y',alpha=alfa))
            elif e.__class__.__name__ == 'octu':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-sh),e.L,2*sh,fc='c',ec='c',alpha=alfa))
            elif e.__class__.__name__ == 'sext':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-sh),e.L,2*sh,fc='r',ec='r',alpha=alfa))
            elif e.__class__.__name__ == 'bend':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-bh),e.L,2*bh,fc='b',ec='b',alpha=alfa))
            elif e.__class__.__name__ in ['kmap','wigg']:
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-kh),e.L,2*kh,fc='g',ec='g',alpha=alfa))
            elif e.__class__.__name__ == 'moni':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-mh),e.L,2*mh,fc='m',ec='m',alpha=alfa))
            elif e.__class__.__name__ == 'kick':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-mc),e.L,2*mc,fc='k',ec='k',alpha=alfa))
            elif e.__class__.__name__ == 'rfca':
                ax.add_patch(mpatches.Rectangle(
                        (s[-1],surflvl-mrf),e.L,2*mrf,fc='r',ec='r',alpha=alfa))
            #elif e.__class__.__name__ == 'sole':
            #    ax.add_patch(mpatches.Rectangle(
            #            (s[-1],surflvl-mrf),e.L,2*kh,fc='c',ec='c',alpha=alfa))
            #elif e.__class__.__name__ == 'mult':
            #    ax.add_patch(mpatches.Rectangle(
            #            (s[-1],surflvl-mrf),e.L,2*mc,fc='k',ec='k',alpha=alfa))
            else:
                print('unknown type: %s'%e.__class__.__name__)
                pass
            s += [s[-1]+e.L]
        plt.plot(s,np.zeros_like(s)+surflvl,'k',linewidth=1.5)


    def plttwiss(self,srange=None,savefn=None,figsize=(15,5),
                 etaxfactor=10,etayfactor=10,surflvl=0):
        '''
        plot twiss and dispersion
        parameters: savefn:    string, if given, save the plot to the file
                    srange:    float/list, a list with two elements to define
                               [s-begin, s-end], whole beam-line by default
        '''
        plt.figure(figsize=figsize)
        plt.plot(self.s,self.betax,'r',linewidth=2,label=r'$\beta_x$')
        plt.plot(self.s,self.betay,'b',linewidth=2,label=r'$\beta_y$')
        plt.plot(self.s,self.etax*etaxfactor,'g',linewidth=2,
                 label=r'$%i\times\eta_x$'%etaxfactor)
        plt.plot(self.s,self.etay*etayfactor,'m',linewidth=2,
                 label=r'$%i\times\eta_y$'%etayfactor)
        self.pltmag(unit=max(self.betax)/20,surflvl=surflvl)
        if srange:
            size = plt.axis()
            plt.axis([srange[0],srange[1],size[2],size[3]])
        plt.xlabel(r'$s$ (m)',fontsize=15)
        plt.ylabel(r'$\beta$ and $\eta$ (m)',fontsize=15)
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=4, mode="expand",borderaxespad=0.)
        if savefn:
            plt.savefig(savefn)
        plt.show()

    def pltsigma(self,srange=None,savefn=None,figsize=(15,5),
                 surflvl=0):
        '''
        plot transverse beam size along s
        parameters: save:      bool, if true, save the plot into the file
                               with the given name as 'fn = xxx.xxx'
                    fn:        str,  specify the file name if save is true
                    srange:    float/list, a list with two elements to define
                               [s-begin, s-end], whole beam-line by default
        '''
        plt.figure(figsize=figsize)
        unit = 1000.
        plt.plot(self.s,np.array(self.sigx)*unit,
                 linewidth=2,label=r'$\sigma_x$')
        plt.plot(self.s,np.array(self.sigy)*unit,
                 linewidth=2,label=r'$\sigma_y$')
        self.pltmag(unit=max(self.sigx)*unit/20,surflvl=surflvl)
        if srange:
            size = plt.axis()
            plt.axis([srange[0],srange[1],size[2],size[3]])
        plt.xlabel(r'$s$ (m)',fontsize=15)
        plt.ylabel(r'$\sigma_x$ (mm)',fontsize=15)
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=2, mode="expand",borderaxespad=0.)
        if savefn:
            plt.savefig(savefn)
        plt.show()

    def savefile(self,fn='savefile.py',comment='your comment',cell=True):
        '''
        save current beamline to a python file
        warning: element information is incompleted
        '''
        ele = list(set(self.bl))
        ele = sorted(ele, key=lambda el: el.__class__.__name__+el.name)
        fid = open(fn,'w')
        if comment:
            fid.write('# --- comments:\n# --- '+str(comment)+'\n\n')
        fid.write('import pylatt as latt\n\n')
        fid.write('# === Element definition:\n')
        for e in ele:
            if e.__class__.__name__ == 'drif':
                s = '{0} = latt.drif("{1}",L={2})\n'.\
                    format(e.name,e.name,e.L)
            elif e.__class__.__name__ == 'bend':
                s = '{0} = latt.bend("{1}",L={2},angle={3},e1={4},e2={5},K1={6},K2={7},hgap={8},fint={9})\n' \
                    .format(e.name,e.name,e.L,e.angle,e.e1,e.e2,e.K1,e.K2,e.hgap,e.fint)
            elif e.__class__.__name__ == 'quad':
                s = '{0} = latt.quad("{1}",L={2},K1={3},tilt={4})\n'.\
                    format(e.name,e.name,e.L,e.K1,e.tilt)
            #elif e.__class__.__name__ == 'sole':
            #    s = '{0} = latt.sole("{1}",L={2},KS={3})\n'.\
            #        format(e.name,e.name,e.L,e.KS)
            #elif e.__class__.__name__ == 'mult':
            #    s = '{0} = latt.mult("{1}",L={2},K1L={3}),K2L={4}\n'.\
            #        format(e.name,e.name,e.L,e.K1L,e.K2L)
            #elif e.__class__.__name__ == 'skew':
            #    s = '{0} = latt.skew("{1}",L={2},K1={3},tilt={4})\n'.\
            #        format(e.name,e.name,e.L,e.K1,e.tilt)
            elif e.__class__.__name__ == 'sext':
                s = '{0} = latt.sext("{1}",L={2},K2={3})\n'.\
                    format(e.name,e.name,e.L,e.K2)
            elif e.__class__.__name__ == 'octu':
                s = '{0} = latt.octu("{1}",L={2},K3={3})\n'.\
                    format(e.name,e.name,e.L,e.K3)
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
            elif e.__class__.__name__ == 'rfca':
                s = '{0} = latt.rfca("{1}",L={2},voltage={3},freq={4})\n'\
                    .format(e.name,e.name,e.L,e.voltage,e.freq)
            elif e.__class__.__name__ == 'wigg':
                s = '{0} = latt.wigg("{1}",L={2},Bw={3})\n'\
                    .format(e.name,e.name,e.L,e.Bw)
            elif e.__class__.__name__ == 'matr':
                fmt = '  [%15.8f,%15.8f,%15.8f,%15.8f,%15.8f,%15.8f],\n'
                m66 = '[\n'
                for mi in range(6):
                    m66 += fmt%tuple(e.tm[mi])
                m66 = m66[:-2] +']\n'
                s = '{0} = latt.matr("{1}",L={2},mat={3})\n'\
                    .format(e.name,e.name,e.L,m66)
            else:
                raise RuntimeError('unknown element type: {0}'.\
                                   format(e.__class__.__name__))
            fid.write(s)
        fid.write('\n# === Beam Line sequence:\n')
        BL = 'BL = ['
        for e in self.bl:
            BL += e.name + ', '
            if len(BL) >= 72:
                fid.write(BL+'\n')
                BL = '   '
        BL = BL[0:-2] + ']\n'
        fid.write(BL)
        if cell:
            fid.write('ring = latt.cell(BL)\n')
        fid.close()

    def saveLte(self,fn='temp.lte',simplify=False):
        '''
        save beamline into ELEGANT format file
        fn:       file's name for saving lattice
        simplify: bool, if True, ignore all drifts with zero length
        warning: some elements need to be adjusted
        '''
        fid = open(fn,'w')
        fid.write('! === Convert from Yongjun Li\'s pylatt input \n')
        fid.write('! === Caution: some elements need to be adjusted manually\n\n')
        fid.write('! === Element definition:\n')
        eset = list(set(self.bl))
        eset = sorted(eset, key=lambda ele: ele.__class__.__name__+ele.name)
        for ei in eset:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                aele = ei.__repr__() \
                    .replace('quad','kquad') \
                    .replace('skew','kquad') \
                    .replace('bend','csbend')\
                    .replace('sext','ksext') \
                    .replace('octu','koct') \
                    .replace('aper','scraper')+'\n'
                fid.write(aele)
        fid.write('\n')
        linehead = '! === Beam line sequence:\nring: LINE = ('
        for ei in self.bl:
            if ei.__class__.__name__=='drif' and ei.L==0 and simplify:
                continue
            else:
                linehead += ei.name+', '
            if len(linehead) >= 72:
                fid.write(linehead+'&\n')
                linehead = '  '
        fid.write(linehead[:-2]+')\n\n')
        fid.write('use, ring\nreturn\n')
        fid.close()

    def getMatLine(self):
        '''
        combine neighboring linear element as matrix list with nonlinear matrix in-between
        return: a list of matr elements with nonlinear elements in-between
        If bends with sext or octu integrated, don't use this function
        '''
        alist = []
        imat = 0
        n = self.getIndex(['sext','octu'])
        # --- 0 -> 1
        if n[0]>0:
            tmlist = [self.bl[j].tm for j in xrange(n[0])][::-1]
            alist.append(matr('M%04i'%imat,L=self.s[n[0]],tm=reduce(np.dot,tmlist)))
            imat += 1
        alist.append(self.bl[n[0]])
        # --- n
        for i,ni in enumerate(n[:-1]):
            tmlist = [self.bl[j].tm for j in xrange(ni+1,n[i+1])][::-1]
            alist.append(matr('M%04i'%imat,L=self.s[n[i+1]]-self.s[ni+1],tm=reduce(np.dot,tmlist)))
            imat += 1
            alist.append(self.bl[n[i+1]])
        # --- n -> end
        if n[-1]<len(self.bl)-1:
            tmlist = [self.bl[j].tm for j in xrange(n[-1]+1,len(self.bl))][::-1]
            alist.append(matr('M%04i'%imat,L=self.L-self.s[n[-1]+1],tm=reduce(np.dot,tmlist)))
        return alist

    def eletrack(self,x0,startIndex=0,endIndex=None,fast=0):
        '''
        element by element track
        x0: 6d initial condition
        endIndex: element index in beamline where tracking stops
                  if None (by default), tracking whole beamline
        '''
        if not fast:
            x0 = np.array(x0,dtype=float).reshape(6,-1)
        if startIndex and startIndex>=0:
            sb = startIndex
        else:
            sb = 0
        if endIndex and endIndex<=len(self.bl):
            se = endIndex
        else:
            se = len(self.bl)
        x = np.zeros((se-sb+1,x0.shape[0],x0.shape[1]))
        x[0] = x0
        for i in xrange(sb,se):
            x0 = self.bl[i].sympass4(x0)
            x[i-sb+1] = x0
        return x

    def getTransMat(self,startIndex=0,endIndex=None):
        '''
        get linear matrix starting from index i0 to i1
        '''
        if startIndex and startIndex>=0:
            sb = startIndex
        else:
            sb = 0
        if endIndex and endIndex<=len(self.bl):
            se = endIndex
        else:
            se = len(self.bl)
        if se > sb:
            tmlist = [e.tm for e in self.bl[sb:se]][::-1]
            R = reduce(np.dot,tmlist)
        elif se < sb:
            tmlist1 = [e.tm for e in self.bl[sb:]]
            tmlist2 = [e.tm for e in self.bl[:se]]
            tmlist = tmlist1+tmlist2
            R = reduce(np.dot,tmlist[::-1])
        else:
            R = np.eye(6)
        return R

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
                if not newline or newline[-1].__class__.__name__!='drif':
                    drfidx += 1
                    dftname = 'd%04i'%drfidx
                    d = drif(dftname,L=ele.L)
                    newline.append(d)
                else:
                    newline[-1].L = newline[-1].L+ele.L
        return newline


class cell(beamline):
    '''
    Periodical solution for a beamline
    '''
    def __init__(self,bl,E=3.,kcouple=0.01):
        '''
        bl:  an element list to compose a beamline
             each elements must be a pre-defined instances of magnet
        E:   float, beam energy in GeV (3 by default)
        kcouple: linear transverse coupling coefficient.
        '''
        self._E = float(E)
        self._bl = [ele for ele in flatten(bl)]
        self._kcouple = kcouple
        self._update()

    @property
    def R(self):
        return self._R

    @property
    def isstable(self):
        return self._isstable

    @property
    def betax0(self):
        return self._betax0

    @property
    def alfax0(self):
        return self._alfax0

    @property
    def betay0(self):
        return self._betay0

    @property
    def alfay0(self):
        return self._alfay0

    @property
    def etax0(self):
        return self._etax0

    @property
    def etaxp0(self):
        return self._etaxp0

    @property
    def emitx(self):
        return self._emitx

    @property
    def emity(self):
        return self._emity

    @property
    def sige(self):
        return self._sige

    @property
    def I(self):
        return self._I

    @property
    def alphac(self):
        return self._alphac

    @property
    def U0(self):
        return self._U0

    @property
    def D(self):
        return self._D

    @property
    def Jx(self):
        return self._Jx

    @property
    def Jy(self):
        return self._Jy

    @property
    def Je(self):
        return self._Je

    @property
    def tau0(self):
        return self._tau0

    @property
    def taux(self):
        return self._taux

    @property
    def tauy(self):
        return self._tauy

    @property
    def taue(self):
        return self._taue

    @property
    def Uw(self):
        return self._Uw

    @property
    def Ut(self):
        return self._Ut

    @property
    def emitxw(self):
        return self._emitxw

    @property
    def sigew(self):
        return self._sigew

    @property
    def tauw(self):
        return self._tauw

    @property
    def kcouple(self):
        return self._kcouple

    @kcouple.setter
    def kcouple(self,value):
        try:
            self._kcouple = float(value)
            self.rad()
        except:
            raise RuntimeError('kcouple must be float (or convertible)')

    def _update(self,rad=False,chrom=False,ndt=False,synosc=False,
                verbose=True):
        '''
        refresh periodical cell parameters
        rad = True, calculate all radiation related parameters
        chrom = True, calculate chromaticity
        ndt = True, calculate nonlinear driving terms
        extraFun: extra function
        extraArg: arguments for the extra function
        verbose: if True, print warning when unstable solution
        '''
        self._spos()
        self._period()
        if self.isstable:
            self._twiss()
            self._disp()
            if rad:
                self.rad()
                self.synosc()
            if chrom:
                self.chrom()
            if ndt: #nonlinear driving terms
                self.geth1()
                self.geth2()
            if synosc:
                self.synosc()

        else:
            if verbose:
                print('\n === Warning: no stable linear optics === \n')
            self.rmattr(['U0', 'sige', 'emitx', 'nux', 'nuy', 'chy',
                         'muy', 'gamay', 'gamax', 'etaxp', 'sigy',
                         'sigx', 'chy0', 'chx0', 'Jy',
                         'etax', 'mux', 'I', 'tau0', 'chx', 'Je',
                         'twx', 'twy', 'D', 'nus',
                         'dx', 'emity', 'Jx', 'taue', 'alphac',
                         'taux', 'tauy', 'alfay', 'alfax',
                         'betay', 'betax'])


    def _period(self):
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
        self._R = self.getTransMat()
        if abs(self.R[0,0]+self.R[1,1])>=2 or \
           abs(self.R[2,2]+self.R[3,3])>=2:
            self._isstable = False
        else:
            mux,self._betax0,self._alfax0,self._gamax0 = matrix2twiss(self.R,plane='x')
            muy,self._betay0,self._alfay0,self._gamay0 = matrix2twiss(self.R,plane='y')
            self._etax0,self._etaxp0,self._etay0,self._etayp0 = matrix2disp(self.R)
            self._isstable = True

    def rmattr(self,attrs):
        '''
        rm attributes, eg. if no stable solution exists, all perodical
        attributes need to be removed
        '''
        for attr in attrs:
            if hasattr(self,attr):
                delattr(self,attr)

    def __repr__(self):
        '''
        List some main parameters
        '''
        s = ''
        s += '\n'+'-'*11*11+'\n'
        s += 11*'%11s'%('s','betax','alfax','mux','etax','etaxp',
                        'betay','alfay','muy','etay','etayp')+'\n'
        s += '-'*11*11+'\n'
        s += (11*'%11.3e'+'\n')%(self.s[0],self.betax[0],self.alfax[0],self.mux[0], \
                                 self.etax[0],self.etaxp[0],self.betay[0],\
                                 self.alfay[0],self.muy[0],self.etay[0],self.etayp[0])
        s += (11*'%11.3e'+'\n')%(self.s[-1],self.betax[-1],self.alfax[-1],self.mux[-1], \
                                 self.etax[-1],self.etaxp[-1],self.betay[-1],\
                                 self.alfay[-1],self.muy[-1],self.etay[0],self.etayp[-1])+'\n'
        if hasattr(self,'nus'):
            s += 'Tune: nux = %8.3f, nuy = %8.3f, nus = %8.3f\n'%(self.nux,self.nuy,self.nus)+'\n'
        else:
            s += 'Tune: nux = %11.3f, nuy = %11.3f\n'%(self.nux,self.nuy)+'\n'
        if hasattr(self,'chx0'):
            s += 'uncorrected chromaticity: chx0 = %11.3e, chy0 = %11.3e\n'%(
                self.chx0,self.chy0)+'\n'
            s += 'corrected chromaticity:   chx  = %11.3e, chy  = %11.3e\n'%(
                self.chx,self.chy)+'\n'
        if hasattr(self,'emitx'):
            s += 'Horizontal emittance [nm.rad] = %11.3e\n'%self.emitx+'\n'
            s += 'Momentum compactor alphac = %11.3e\n'%self.alphac+'\n'
            s += 'Radiation damping factor: D = %11.3e\n'%self.D+'\n'
            s += 'Radiation damping partition factors: Jx = %11.3e, Jy = %11.3e, Je = %11.3e\n'\
                 %(self.Jx,self.Jy,self.Je)+'\n'
            s += 'Radiation damping time [ms]: taux = %11.3e, tauy = %11.3e, taue = %11.3e\n'\
                 %(self.taux,self.tauy,self.taue)+'\n'
            s += 'Radiation loss per turn U0 [keV] = %11.3e\n'%self.U0+'\n'
            s += 'fractional energy spread sige = %11.3e\n'%self.sige+'\n'
            s += 'synchrotron radiation integrals\n'
            s += ('-'*5*11+'\n'+5*'%11s')%('I1','I2','I3','I4','I5')+'\n'
            s += ('-'*5*11+'\n'+5*'%11.3e'+'\n')%(self.I[0],self.I[1],self.I[2],
                                                  self.I[3],self.I[4])+'\n'
        if hasattr(self,'h1') or hasattr(self,'h2'):
            s += self.showh12()
        s += '\n\n'
        return s

    def rad(self,includeWig=0):
        '''
        copy from S. Krinsky's code:
        Radiation integrals calculated according to Helm, Lee and Morton,
        IEEE Trans. Nucl. Sc., NS-20, 1973, p900
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
        const = 55./(32.*np.sqrt(3))
        re = 2.817940e-15 #unit:  m
        lam = 3.861592e-13 #unit: m
        grel = 1956.95*self.E #E in GeV
        Ee = 0.510999 #unit: MeV
        self._I = 5*[0.]
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
                b0 = self.betax[i]
                a0 = self.alfax[i]
                eta0 = self.etax[i]
                etap0 = self.etaxp[i]
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
                           (2.*angle)*(-(g1*eta0+a1*etap1)*(p-S)/(k*p**2) + \
                                       (a1*eta0+b0*etap1)*(1-C)/p**2) + \
                           angle**2*(g1*(3.*p-4.*S+S*C)/(2.*k**2*p**3) - \
                                     a1*(1.-C)**2/(k*p**3)+b0*(p-C*S)/(2.*p**3))
                elif K+1/R**2 < 0:
                    k = np.sqrt(-K-1/R**2)
                    p = k*L
                    S = np.sinh(p)
                    C = np.cosh(p)
                    eta2 = eta0*C+etap1*S/k+(1-C)/(-R*k**2)
                    av_eta = eta0*S/p+etap1*(1-C)/(-p*k)+(p-S)/(-p*k**2*R)
                    av2 = -K*av_eta/R+(eta0*t1+eta2*t2)/(2.*L*R)
                    av_H = g1*eta0**2+2*a1*eta0*etap1+b0*etap1**2 + \
                           (2.*angle)*(-(g1*eta0+a1*etap1)*(p-S)/(-k*p**2) + \
                                    (a1*eta0+b0*etap1)*(1-C)*(-1.)/p**2) + \
                           angle**2*(g1*(3.*p-4.*S+S*C)/(2.*k**2*p**3) - \
                                     a1*(1.-C)**2/(k*p**3)+b0*(p-C*S)/(-2.*p**3))
                else:
                    print('1/R**2+K1 == 0, not implemented')
                    pass
                self._I[0] += L*av_eta/R
                self._I[1] += L/R**2
                self._I[2] += L/abs(R)**3
                self._I[3] += L*av_eta/R**3-2.*L*av2
                self._I[4] += L*av_H/abs(R)**3
        self._alphac = self.I[0]/self.L
        self._U0 = 666.66*re*Ee*grel**4*self.I[1]
        self._D = self.I[3]/self.I[1]
        self._Jx = 1-self.D
        self._Jy = 1.
        self._Je = 2+self.D
        self._tau0 = 3*self.L*1e3/(re*csp*grel**3*self.I[1])
        self._taux = self.tau0/self.Jx
        self._tauy = self.tau0
        self._taue = self.tau0/self.Je
        self._sige = np.sqrt(const*lam*grel**2*self.I[2]/(2.*self.I[1]+self.I[3]))
        self._emitx = const*lam*grel**2*(self.I[4]/(self.I[1]-self.I[3]))*1e9 #nm
        self._emity = self.emitx*self.kcouple/(1+self.kcouple) #nm
        self._sigx =[np.sqrt(self.betax[m]*self.emitx*1e-9) +
                     abs(self.etax[m])*self.sige for m in range(len(self.s))]
        self._sigy =[np.sqrt(self.betay[m]*self.emity*1e-9)
                     for m in range(len(self.s))]
        if includeWig:
            # --- S.Y. Lee's book formulae
            Uw,u3,u2 = 0,0,0
            dwIndex = self.getIndex('kmap',prefix='dw',exitport=0)
            E0 = self.bl[dwIndex[0]].E
            rho = 88.5*E0**4/self.U0
            for i in dwIndex:
                dw = self.bl[i]
                rhow = dw.E*1e9/csp/dw.Bw
                Uwi = 88.5*E0**4*dw.L/(2*twopi*rhow**2)
                Uw += Uwi
                u3 += 8*rho*Uwi/3/np.pi/rhow/self.U0
                u2 += Uwi/self.U0
            self.Uw = Uw
            self.Ut = self.U0+self.Uw
            self.emitxw = self.emitx/(1+self.Uw/self.U0)
            self.sigew = self.sige*np.sqrt((1.+u3)/(1.+u2))
            self.tauw = self.tau0*self.U0/self.Ut
            self.tauxw = self.tauw/self.Jx
            self.tauyw = self.tauw
            self.tauew = self.tauw/self.Je

    def findClosedOrbit(self,niter=50,fixedenergy=0,
                        tol=[1e-6,1e-7,1e-6,1e-7,1e-6,1e-6],
                        sym4=True,verbose=False):
        '''
        find closed orbit by iteration
        niter: number of iterations
        tol: tolerance
        so far, logitudinally, only for fixed energy
        returns: xco,xpco,yco,ypco,isconvergent, and dl (path-length change)
        '''
        isconvergent = False
        x1all = None
        x0 = np.zeros(6)
        x0[5] = fixedenergy
        for i in range(niter):
            x1all = self.eletrack(x0)
            x1 = x1all[-1][:,0]
            if abs(x0[0]-x1[0])<=tol[0] and \
               abs(x0[1]-x1[1])<=tol[1] and \
               abs(x0[2]-x1[2])<=tol[2] and \
               abs(x0[3]-x1[3])<=tol[3]:
                isconvergent = True
                break
            else:
                x0 = (x0+x1)/2
                x0[4] = 0.
        if isconvergent:
            xco = x1all[:,0,:][:,0]
            xpco = x1all[:,1,:][:,0]
            yco = x1all[:,2,:][:,0]
            ypco = x1all[:,3,:][:,0]
            if verbose:
                print('closed orbit found after %i iterations'%(i+1))
            return xco,xpco,yco,ypco,isconvergent,x1all[-1,4,0]
        else:
            xco = x1all[:,0,:][:,0]
            yco = x1all[:,2,:][:,0]
            if verbose:
                print('max. iteration (%i) reached, no closed orbit found'%niter)
            return xco,xpco,yco,ypco,isconvergent,x1all[-1,4,0]


    def dispersionTrack(self,dE=np.linspace(-0.025,0.025,8),deg=3,mp=True,
                        verbose=False,figsize=(15,9)):
        '''
        get higher order dispersion and momentum compactor using
        symplectic tracking
        dE:      delta list
        deg:     polyfit order
        mp:      multiprocessing (fast)
        dispx/y: higher order dispersion
        alpha:   higher order momentom compactor
        '''
        if mp:
            def f(q,de):
                xco,xpco,yco,ypco,c,dl = self.findClosedOrbit(
                    fixedenergy=de,sym4=True,niter=100)
                q.put([de,xco,yco,dl])

            def handler(dE):
                q = [Queue() for i in range(len(dE))]
                p = [Process(target=f,args=(qi,de)) for qi,de in zip(q,dE)]
                [pi.start() for pi in p]
                result = [qi.get() for qi in q]
                for qi in q:
                    qi.close()
                    qi.join_thread()
                [pi.join() for pi in p]
                return result

            result = handler(dE)
            e,x,y,dL = [],[],[],[]
            for a in result:
                e.append(a[0])
                x.append(a[1])
                y.append(a[2])
                dL.append(a[3])
            idx = np.argsort(e)
            e = np.array(e)[idx]
            x = np.array(x)[idx]
            y = np.array(y)[idx]
            dL = np.array(dL)[idx]
            self.dispx = np.polyfit(e,x,deg)
            self.dispy = np.polyfit(e,y,deg)
            self.alpha = np.polyfit(e,dL/self.L,deg)
        else:
            Xco,Yco,C,dL = [],[],[],[]
            for i,de in enumerate(dE):
                if verbose:
                    sys.stdout.write('\r %03i out of %03i: dE = %9.4f'
                                     %(i+1,len(dE),de))
                    sys.stdout.flush()
                xco,xpco,yco,ypco,c,dl = self.findClosedOrbit(fixedenergy=de,
                                            sym4=True,niter=100)
                Xco.append(xco)
                Yco.append(yco)
                C.append(c)
                dL.append(dl)
            Xco = np.array(Xco)
            Yco = np.array(Yco)
            dL = np.array(dL)
            self.dispx = np.polyfit(dE,Xco,deg=deg)
            self.dispy = np.polyfit(dE,Yco,deg=deg)
            self.alpha = np.polyfit(dE,dL/self.L,deg=deg)
        if verbose:
            plt.figure(figsize=figsize)
            for i,order in enumerate(range(-2,-deg-1,-1)):
                plt.subplot(deg-1,1,i+1)
                plt.plot(self.s,self.dispx[order],'-',linewidth=2,
                         label=r'$\eta_{x'+str(i+1)+'}$')
                if i == 0:
                    plt.plot(self.s,self.etax,'o',label='Linear theory')
                plt.xlabel('s (m)')
                plt.ylabel(r'$\eta_x (m)$')
                plt.legend(loc='best')
            plt.show()

    def chromTrack(self,dE=np.linspace(-.025,0.025,16),
                   nturn=516,amp=1e-4,deg=4,tunewindow=0,
                   crossHalfInt=0,verbose=False,figsize=(8,6)):
        '''
        get chromaticity by tracking
        '''
        #de = np.linspace(demin,demax,int(nde))
        nde = len(dE)
        xin = np.zeros((6,nde))
        #if plane == 'x':
        xin[0] = nde*[amp]
        #else:
        xin[2] = nde*[amp]
        xin[5] = dE
        tbt = np.zeros((nturn,6,nde))
        for i in xrange(nturn):
            xin = self.eletrack(xin)[-1]
            tbt[i] = xin
            sys.stdout.write(
                '\r--- tracking: %04i out of %04i is being done (%3i%%) ---'
                %(i+1,nturn,(i+1)*100./nturn))
            sys.stdout.flush()
        nu = []
        if tunewindow:
            rngx = [self.nux-int(self.nux)-tunewindow,self.nux-int(self.nux)+tunewindow]
            rngy = [self.nuy-int(self.nuy)-tunewindow,self.nuy-int(self.nuy)+tunewindow]
        else:
            rngx,rngy = [0.,1],[0.,1]
        if crossHalfInt:
            csm = CSNormalization(self.betax[0],self.alfax[0],
                                  self.betay[0],self.alfay[0],combine=True)
        for i in xrange(nde):
            if crossHalfInt:
                a0 = np.array(csm*tbt[:,:4,i].transpose())
                xpw0 = np.fft.fft(a0[0])
                nu1 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                xpw0 = np.fft.fft(a0[1])
                nu2 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                nu.append([dE[i],nu1,nu2])
            else:
                nu.append([dE[i],naff(tbt[:,0,i],ni=1,verbose=0,rng=rngx)[0][0],
                           naff(tbt[:,2,i],ni=1,verbose=0,rng=rngy)[0][0]])
        nu = np.array(nu)
        chx = np.polyfit(nu[:,0],nu[:,1],deg)
        chy = np.polyfit(nu[:,0],nu[:,2],deg)
        if verbose:
            nuxfit = np.polyval(chx,nu[:,0])
            nuyfit = np.polyval(chy,nu[:,0])
            plt.figure(figsize=figsize)
            plt.plot(nu[:,0],nuxfit,'b-',label=r"$\xi_{x,1}=%.2f$"%chx[-2],lw=2)
            plt.plot(nu[:,0],nu[:,1],'bo')
            plt.plot(nu[:,0],nuyfit,'r-',label=r"$\xi_{y,1}=%.2f$"%chy[-2],lw=2)
            plt.plot(nu[:,0],nu[:,2],'ro')
            plt.xlabel(r'$\delta = \frac{\Delta P}{P}$',fontsize=18)
            plt.ylabel(r'$\nu$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            plt.show()
        self.chxTrack,self.chyTrack,self.chromTbt = chx,chy,tbt

    # --- linear coupling
    def coupledTwiss(self):
        '''
        calculated Twiss functions (four Betas)  with linear coupling
        twx(y)_I(II): beta-alfa-gama-phase
        '''
        l0,v0 = np.linalg.eig(self.R[:4,:4])
        self.nu_I,self.nu_II = np.arctan2(l0[0].imag,l0[0].real)/twopi,\
                               np.arctan2(l0[2].imag,l0[2].real)/twopi
        bag,bagName = vect2beta(v0)
        bagx_I,bagy_I,bagx_II,bagy_II = [bag[:,0]],[bag[:,1]],[bag[:,2]],[bag[:,3]]
        for ele in self.bl:
            v1 = ele.tm[:4,:4].dot(v0)
            t,tn = vect2beta(v1)
            bagx_I.append(t[:,0])
            bagy_I.append(t[:,1])
            bagx_II.append(t[:,2])
            bagy_II.append(t[:,3])
            v0 = v1
        self.twx_I,self.twy_I = np.array(bagx_I).transpose(),np.array(bagy_I).transpose()
        self.twx_II,self.twy_II = np.array(bagx_II).transpose(),np.array(bagy_II).transpose()
        self.twx_I[3] = monoPhase(self.twx_I[3])
        self.twy_I[3] = monoPhase(self.twy_I[3])
        self.twx_II[3] = monoPhase(self.twx_II[3])
        self.twy_II[3] = monoPhase(self.twy_II[3])

    def coupledemit(self):
        '''
        calculated Twiss functions (four Betas)  with linear coupling
        twx(y)_I(II): beta-alfa-gama-phase
        '''
        Cq = 3.832e-13
        gamma = self.E*1e3/0.5109989461
        l0,v0 = np.linalg.eig(self.R[:4,:4])
        Hx_I,Hx_II,Hy_I,Hy_II = 0,0,0,0
        I2,I4_I,I4_II = 0,0,0
        for i,ele in enumerate(self.bl):
            if ele.__class__.__name__ != 'bend':
                v1 = ele.tm[:4,:4].dot(v0)
                v0 = v1
            else:
                s,w = gint(0.,ele.L)
                b,a,g = [],[],[]
                dx,dxp,dy,dyp = [],[],[],[]
                for si in s:
                    if si == ele.L:
                        fe = bend(L=ele.L,K1=ele.K1,angle=ele.angle,
                                  e1=ele.e1,e2=ele.e2)
                    elif si < ele.L:
                        fe = bend(L=si,K1=ele.K1,angle=ele.angle*si/ele.L,e1=ele.e1)
                    vm = fe.tm[:4,:4].dot(v0)
                    t = np.array(vect2beta(vm)[0])
                    # --- twiss inside bend
                    b.append(t[0])
                    a.append(t[1])
                    g.append(t[2])
                    # --- dispersion
                    mxy,tx,ty = twmat(ele,si)
                    dxy = mxy.dot(self._dxy[:,i])
                    dx.append(dxy[0])
                    dxp.append(dxy[1])
                    dy.append(dxy[2])
                    dyp.append(dxy[3])
                b = np.array(b)
                a = np.array(a)
                g = np.array(g)
                bx_I,ax_I,gx_I = b[:,0],a[:,0],g[:,0]
                by_I,ay_I,gy_I = b[:,1],a[:,1],g[:,1]
                bx_II,ax_II,gx_II = b[:,2],a[:,2],g[:,2]
                by_II,ay_II,gy_II = b[:,3],a[:,3],g[:,3]
                I4_I += sum(w*dx/ele.R*(1/ele.R**2+2*ele.K1))
                I4_II += sum(w*dy/ele.R*(1/ele.R**2+2*ele.K1))
                Hx_I += sum(w*(gx_I*dx*dx+2*ax_I*dx*dxp+bx_I*dxp*dxp)/ele.R**3)
                Hy_I += sum(w*(gy_I*dy*dy+2*ay_I*dy*dyp+by_I*dyp*dyp)/ele.R**3)
                Hx_II += sum(w*(gx_II*dx*dx+2*ax_II*dx*dxp+bx_II*dxp*dxp)/ele.R**3)
                Hy_II += sum(w*(gy_II*dy*dy+2*ay_II*dy*dyp+by_II*dyp*dyp)/ele.R**3)
                I2 += ele.L/ele.R**2
                v1 = ele.tm[:4,:4].dot(v0)
                v0 = v1
        #print "x_I,x_II: ",Hx_I/(I2-I4_I)*Cq*gamma**2*1e9,Hx_II/(I2-I4_II)*Cq*gamma**2*1e9
        #print "y_I,y_II: ",Hy_I/(I2-I4_I)*Cq*gamma**2*1e9,Hy_II/(I2-I4_II)*Cq*gamma**2*1e9
        #self.emitx_c = Hx_I/(I2-I4_I)*Cq*gamma**2*1e9+Hx_II/(I2-I4_II)*Cq*gamma**2*1e9
        #self.emity_c = Hy_I/(I2-I4_I)*Cq*gamma**2*1e9+Hy_II/(I2-I4_II)*Cq*gamma**2*1e9
        self.emit_I = (Hx_I/(I2-I4_I)+Hy_I/(I2-I4_I))*Cq*gamma**2*1e9
        self.emit_II = (Hx_II/(I2-I4_II)+Hy_II/(I2-I4_II))*Cq*gamma**2*1e9

    def pltcoupledtwiss(self,figsize=(15,5),savefn=None,lw=2,surflvl=0):
        '''
        plot coupled beta-functions
        '''
        if not hasattr(self,'twx_I'):
            self.coupledTwiss()
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.plot(self.s,self.twx_I[0],'b',label=r'$\beta_{x,I}$',lw=lw)
        plt.plot(self.s,self.twy_II[0],'r',label=r'$\beta_{y,II}$',lw=lw)
        self.pltmag(unit=max(self.twx_I[0])/20,surflvl=surflvl)
        plt.xlabel('s (m)',fontsize=15)
        plt.ylabel(r'$\beta (m)$',fontsize=15)
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=2, mode="expand",borderaxespad=0.)
        plt.subplot(212)
        plt.plot(self.s,self.twx_II[0],'b',label=r'$\beta_{x,II}$',lw=lw)
        plt.plot(self.s,self.twy_I[0],'r',label=r'$\beta_{y,I}$',lw=lw)
        self.pltmag(unit=max(self.twx_II[0])/20,surflvl=surflvl)
        plt.xlabel('s (m)',fontsize=15)
        plt.ylabel(r'$\beta (m)$',fontsize=15)
        plt.legend(bbox_to_anchor=(0, 1.005, 1, .1), loc=3,
                   ncol=2, mode="expand",borderaxespad=0.)
        if savefn:
            plt.savefig(savefn)
        plt.show()

    def getOneTurnMatrix(self,idx=0):
        '''
        get one turn linear matrix starting from given index, 0 by default
        '''
        tmlist1 = [e.tm for e in self.bl[idx:]]
        tmlist2 = [e.tm for e in self.bl[:idx]]
        tmlist = tmlist1+tmlist2
        return reduce(np.dot,tmlist[::-1])

    def getf2vsskew(self,index,skews,dk=0.001,verbose=0):
        '''
        coupled f2 coefficients vs. skew response matrix
        '''
        nbpm,nskew = len(index),len(skews)
        c1010,c0101 = np.zeros((nbpm,nskew)),np.zeros((nbpm,nskew))
        c0110,c1001 = np.zeros((nbpm,nskew)),np.zeros((nbpm,nskew))
        h1010r,h1010i = np.zeros((nbpm,nskew)),np.zeros((nbpm,nskew))
        h1001r,h1001i = np.zeros((nbpm,nskew)),np.zeros((nbpm,nskew))
        for ik,skew in enumerate(skews):
            skew.K1 = dk
            self._update()
            cc1010,cc0101,cc0110,cc1001 = [],[],[],[]
            hh1010r,hh1010i,hh1001r,hh1001i = [],[],[],[]
            for bi in index:
                Ri = self.getOneTurnMatrix(bi)
                f2 = tm2f2(Ri)
                ih1010r,ih1010i,ih1001r,ih1001i = f22h2(f2)
                cc1010.append(f2.pickWithIndex([1,0,1,0])/dk)
                cc0101.append(f2.pickWithIndex([0,1,0,1])/dk)
                cc0110.append(f2.pickWithIndex([0,1,1,0])/dk)
                cc1001.append(f2.pickWithIndex([1,0,0,1])/dk)
                hh1010r.append(ih1010r/dk)
                hh1010i.append(ih1010i/dk)
                hh1001r.append(ih1001r/dk)
                hh1001i.append(ih1001i/dk)
            skew.K1 = 0
            c1010[:,ik] = cc1010
            c0101[:,ik] = cc0101
            c0110[:,ik] = cc0110
            c1001[:,ik] = cc1001
            h1010r[:,ik] = hh1010r
            h1010i[:,ik] = hh1010i
            h1001r[:,ik] = hh1001r
            h1001i[:,ik] = hh1001i
            if verbose:
                sys.stdout.write('\r --- %04i out of %04i done (%3i%%) ---'\
                                 %(ik+1,nskew,(ik+1)*100./nskew))
                sys.stdout.flush()
        self.c1010rm = c1010
        self.c0101rm = c0101
        self.c0110rm = c0110
        self.c1001rm = c1001
        self.h1010rrm = h1010r
        self.h1010irm = h1010i
        self.h1001rrm = h1001r
        self.h1001irm = h1001i
        self._update()

    def getCouplingResponseMatrix(self,index,skews,dk=0.001,verbose=0):
        '''
        one turn matrix vs. skew response matrix
        '''
        R0 = []
        for i in index:
            Rt = np.array(self.getOneTurnMatrix(i)[:4,:4])
            Ri = np.vstack((Rt[:2,2:],Rt[2:,:2])).flatten()
            R0.append(Ri)
        R0 = np.array(R0).flatten()
        nskew = len(skews)
        rm = np.zeros((len(R0),nskew))
        for ik,skew in enumerate(skews):
            skew.K1 = dk
            self._update()
            R1 = []
            for i in index:
                Rt = np.array(self.getOneTurnMatrix(i)[:4,:4])
                Ri = np.vstack((Rt[:2,2:],Rt[2:,:2])).flatten()
                R1.append(Ri)
            R1 = np.array(R1).flatten()
            dRdk = (R1-R0)/dk
            rm[:,ik] = dRdk
            skew.K1 = 0
            if verbose:
                sys.stdout.write('\r --- %04i out of %04i done (%3i%%) ---'\
                                 %(ik+1,nskew,(ik+1)*100./nskew))
                sys.stdout.flush()
        self.lcrm = rm
        self._update()
        return rm,R0
    # --- end of linear coupling

    # --- synchrotron oscillation
    def synosc(self,verbose=False):
        '''
        synchrotron oscillation
        '''
        if not hasattr(self,'U0'):
            self.rad()
        rfcas = self.getElements('rfca')
        if len(rfcas) == 0:
            if verbose:
                print('=== warning: no RF cavity found ===')
            return
        vt = 0.
        for rfca in rfcas:
            vt += rfca.voltage
            freqrf = rfca.freq
        self.Vrf = vt
        if hasattr(self,'Ut'):
            self.phaserf = np.pi-np.arcsin(self.Ut*1e3/self.Vrf)
        else:
            self.phaserf = np.pi-np.arcsin(self.U0*1e3/self.Vrf)
        self.revfreq = csp/self.L
        self.h = int(round(freqrf/self.revfreq))
        self.freqrf = self.h*self.revfreq
        for rfca in rfcas:
            rfca.freq = self.freqrf
        self.nus = np.sqrt(self.h*self.Vrf \
                           *abs(np.cos(self.phaserf)*self.alphac)/ \
                           (twopi*self.E*1e9))
        # --- bucket height
        Y = np.sqrt(abs(np.cos(self.phaserf) - \
                        (np.pi/2-self.phaserf)*np.sin(self.phaserf)))
        self.bucketHeight = np.sqrt(2.*self.Vrf/ \
                                    (self.E*1e9*np.pi*self.h*abs(self.alphac)))

    def synHam(self,dE,phi,ho=False):
        '''
        calculate synchrotron oscillation Hamiltonian with
        given deltaE and phase.
        if ho (higher order) is true, using up to 2nd alphac
        to calculate Hamiltonian
        '''
        if not ho:
            H = 1./2*self.h*self.revfreq*self.alphac*dE**2 + \
            self.revfreq*self.Vrf/(2*np.pi*self.E*1e9) * \
            (np.cos(phi)-np.cos(self.phaserf) +
             (phi-self.phaserf)*np.sin(self.phaserf))
        else:
            if not hasattr(self,'alpha'):
                self.dispersionTrack()
            H = 1./2*self.h*self.revfreq*self.hoalpha[-2]*dE**2 + \
                1./3*self.h*self.revfreq*self.hoalpha[-3]*dE**3 + \
                1./4*self.h*self.revfreq*self.hoalpha[-4]*dE**4 + \
                self.revfreq*self.Vrf/(2*np.pi*self.E*1e9) * \
                (np.cos(phi)-np.cos(self.phaserf) +
                 (phi-self.phaserf)*np.sin(self.phaserf))
        return H

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
            if elem.__class__.__name__ in ['drif','moni','aper','wigg','matr','sole']:
                continue
            if elem.__class__.__name__ == 'mult':
                dchx = -elem.K1L*self.betax[i]
                dchy =  elem.K1L*self.betay[i]
                chx0 += dchx
                chy0 += dchy
                chx += dchx
                chy += dchy
                if cor:
                    dchx =  elem.K2L*self.etax[i]*self.betax[i]
                    dchy = -elem.K2L*self.etax[i]*self.betay[i]
                    chx += dchx
                    chy += dchy
                continue
            s,w = gint(0.,elem.L)
            betax,alphax,gammax = [],[],[]
            betay,alphay,gammay = [],[],[]
            eta,etap=[],[]
            for j in range(8):
                mx,my,tx,ty = twmat(elem,s[j])
                wx = twisstrans(tx,self.twx[:,i])
                wy = twisstrans(ty,self.twy[:,i])
                dx = mx.dot(np.take(self.dxy[:,i],[0,1,4],axis=0))
                betax.append(wx[0])
                alphax.append(wx[1])
                gammax.append(wx[2])
                betay.append(wy[0])
                alphay.append(wy[1])
                gammay.append(wy[2])
                eta.append(dx[0])
                etap.append(dx[1])
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
                    dchx = -h**2*sum(w*betax)-2*h*sum(w*etap*alphax) + \
                           h*sum(w*eta*gammax)
                    dchy = h*sum(w*eta*gammay)
                else:
                    dchx = -(elem.K1+h**2)*sum(w*betax) - \
                           2*elem.K1*h*sum(w*eta*betax) - \
                           2*h*sum(w*etap*alphax)+h*sum(w*eta*gammax)
                    dchy = elem.K1*sum(w*betay)-h*elem.K1*sum(w*eta*betay)+\
                           h*sum(w*eta*gammay)
                if hasattr(elem,'K2'):
                    dchx += elem.K2*sum(w*eta*betax)
                    dchy -= elem.K2*sum(w*eta*betay)
                b0x = self.twx[0,i]
                b0y = self.twy[0,i]
                b1x = self.twx[0,i+1]
                b1y = self.twy[0,i+1]
                dchx += h*b0x*np.tan(elem.e1)+h*b1x*np.tan(elem.e2)
                dchy -= h*b0y*np.tan(elem.e1)+h*b1y*np.tan(elem.e2)
                chx += dchx
                chy += dchy
                chx0 += dchx
                chy0 += dchy
            elif elem.__class__.__name__ == 'sext' and cor:
                chx += elem.K2*sum(w*eta*betax)
                chy -= elem.K2*sum(w*eta*betay)
        self.chx = chx/(2*twopi)*self.N
        self.chy = chy/(2*twopi)*self.N
        self.chx0 = chx0/(2*twopi)*self.N
        self.chy0 = chy0/(2*twopi)*self.N

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


    def cchrom1(self,var,ch=[0,0]):
        '''
        use specified sexts to correct chromaticity
        if minSextValue is True, try to minimize the
        sum of absolute sextupole strengthes
        '''
        if not hasattr(self,'chrm'):
            self.getChrm(self,var)
        self.chrom()
        dk = np.linalg.lstsq(self.chrm,
                             np.array(ch)-np.array([self.chx,self.chy]))[0]
        for i,v in enumerate(var):
            v.put('K2',v.K2+dk[i])
        self.chrom()


    def simuTbt(self,bpmIndex=[],nturns=1050,x0=[1.e-3,0,0,0,0,0],verbose=0):
        '''
        simulate TbT
        '''
        if not len(bpmIndex):
            bpmIndex = ring.getIndex('moni')
        x = np.zeros((len(bpmIndex),nturns))
        y = np.zeros((len(bpmIndex),nturns))
        for i in range(nturns):
            xall = self.eletrack(x0)
            x[:,i] = xall[bpmIndex,0,0]
            y[:,i] = xall[bpmIndex,2,0]
            x0 = xall[-1]
            if verbose:
                sys.stdout.write('\r --- %04i out of %04i done (%3i%%) ---'\
                                 %(i+1,nturns,(i+1)*100./nturns))
                sys.stdout.flush()
        return x,y


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
        dx = mx*np.take(self.dxy[:,i],[0,1,4],axis=0)
        bxi,byi = twx[0,0],twy[0,0]
        dxi = dx[0,0]
        if ei.L < 0:
            neglen = True
        else:
            neglen = False
        dpx = phasetrans(mx,self.twx[:,i],neglen=neglen)
        dpy = phasetrans(my,self.twy[:,i],neglen=neglen)
        muxi,muyi = self.mux[i]+dpx,self.muy[i]+dpy
        return bxi,byi,dxi,muxi*twopi,muyi*twopi


    def createh12table(self):
        '''
        create twiss table for h1 and h2 computation
        '''
        tab = []
        mags = self.getElements(['quad','sext'])
        magIndexs = self.getIndex(['quad','sext'],exitport=0)
        for i,mag in enumerate(mags):
            if mag.__class__.__name__ == 'quad':
                pl = [1]
                bn = mag.K1
            else: # --- sext
                pl = [2]
                bn = mag.K2/2
            bx,by,dx,mx,my = self.twmid(magIndexs[i])
            pl += [mag.L,bn,bx,by,dx,mx,my]
            tab.append(pl)
        self.h12table = np.array(tab)


    def geth1(self):
        '''
        non-linear driving terms
        '''
        if not hasattr(self,'h12table'):
            self.createh12table()

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

        for i,ei in enumerate(self.h12table):
            if ei[0] == 2:
                Li,bni,bxi,byi,dxi,muxi,muyi = ei[1],ei[2],ei[3],ei[4],ei[5],ei[6],ei[7]
                K2Li = Li*bni
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

            if ei[0] == 1:
                Li,bni,bxi,byi,dxi,muxi,muyi = ei[1],ei[2],ei[3],ei[4],ei[5],ei[6],ei[7]
                K1Li = Li*bni
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)

            # --- dipole fringe field was ignored

        self.h1 = {'h11001':h11001/4,'h00111':h00111/4,'h20001':h20001/8,'h00201':h00201/8,
                   'h10002':h10002/2,'h21000':h21000/8,'h30000':h30000/24,'h10110':h10110/4,
                   'h10020':h10020/8,'h10200':h10200/8}


    def geth2(self,**kwargs):
        '''
        non-linear driving terms
        ref: Second-order driving terms due to sextupoles and chromatic effects of quadrupoles
        AOP-TN-2009-020
        Chun-xi Wang
        '''
        if not hasattr(self,'h12table'):
            self.createh12table()

        if kwargs.get('all',0):
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
        else:
            if kwargs.get('h22000',0):
                h22000 = complex(0,0)
            if kwargs.get('h31000',0):
                h31000 = complex(0,0)
            if kwargs.get('h11110',0):
                h11110 = complex(0,0)
            if kwargs.get('h11200',0):
                h11200 = complex(0,0)
            if kwargs.get('h40000',0):
                h40000 = complex(0,0)
            if kwargs.get('h20020',0):
                h20020 = complex(0,0)
            if kwargs.get('h20110',0):
                h20110 = complex(0,0)
            if kwargs.get('h20200',0):
                h20200 = complex(0,0)
            if kwargs.get('h00220',0):
                h00220 = complex(0,0)
            if kwargs.get('h00310',0):
                h00310 = complex(0,0)
            if kwargs.get('h00400',0):
                h00400 = complex(0,0)
            if kwargs.get('h21001',0):
                h21001 = complex(0,0)
            if kwargs.get('h30001',0):
                h30001 = complex(0,0)
            if kwargs.get('h10021',0):
                h10021 = complex(0,0)
            if kwargs.get('h20002',0):
                h20002 = complex(0,0)
            if kwargs.get('h00112',0):
                h00112 = complex(0,0)
            if kwargs.get('h00202',0):
                h00202 = complex(0,0)
            if kwargs.get('h10003',0):
                h10003 = complex(0,0)
            if kwargs.get('h00004',0):
                h00004 = complex(0,0)
            if kwargs.get('h',0):
                h = complex(0,0)
            if kwargs.get('h',0):
                h = complex(0,0)

        for i,ei in enumerate(self.h12table):
            if ei[0] == 2:
                Li,bni,bxi,byi,dxi,muxi,muyi = ei[1],ei[2],ei[3],ei[4],ei[5],ei[6],ei[7]
                K2Li = Li*bni
                for j,ej in enumerate(self.h12table):
                    if ej[0] == 2:
                        Lj,bnj,bxj,byj,dxj,muxj,muyj = ej[1],ej[2],ej[3],ej[4],ej[5],ej[6],ej[7]
                        K2Lj = Lj*bnj
                        if i != j:
                            if 'h22000' in locals(): d22000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*(np.exp(3j*(muxi-muxj))+\
                                                                      3*np.exp(1j*(muxi-muxj)))
                            if 'h31000' in locals(): d31000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*np.exp(1j*(3*muxi-muxj))
                            if 'h11110' in locals(): d11110 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-muxj))-\
                                                                          np.exp(1j*(muxi-muxj)))+\
                                                                     byj*(np.exp(1j*(muxi-muxj+2*muyi-2*muyj))+\
                                                                          np.exp(-1j*(muxi-muxj-2*muyi+2*muyj))))
                            if 'h11200' in locals(): d11200 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-muxj-2*muyi))-\
                                                                          np.exp(1j*(muxi-muxj+2*muyi)))+\
                                                                   2*byj*(np.exp(1j*(muxi-muxj+2*muyi))+\
                                                                          np.exp(-1j*(muxi-muxj-2*muyj))))
                            if 'h40000' in locals(): d40000 = K2Li*K2Lj*bxi**(3./2)*bxj**(3./2)*np.exp(1j*(3*muxi+muxj))
                            if 'h20020' in locals(): d20020 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*np.exp(-1j*(muxi-3*muxj+2*muyi))-\
                                                             (bxj+4*byj)*np.exp(1j*(muxi+muxj-2*muyi)))
                            if 'h20110' in locals(): d20110 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*(np.exp(-1j*(muxi-3*muxj))-np.exp(1j*(muxi+muxj)))+\
                                                                    2*byj*np.exp(1j*(muxi+muxj+2*muyi-2*muyj)))
                            if 'h20200' in locals(): d20200 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*(bxj*np.exp(-1j*(muxi-3*muxj-2*muyi))-\
                                                             (bxj-4*byj)*np.exp(1j*(muxi+muxj+2*muyi)))
                            if 'h00220' in locals(): d00220 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*(np.exp(1j*(muxi-muxj+2*muyi-2*muyj))+\
                                                                       4*np.exp(1j*(muxi-muxj))-\
                                                                         np.exp(-1j*(muxi-muxj-2*muyi+2*muyj)))
                            if 'h00310' in locals(): d00310 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*(np.exp(1j*(muxi-muxj+2*muyi))-\
                                                                         np.exp(-1j*(muxi-muxj-2*muyi)))
                            if 'h00400' in locals(): d00400 = K2Li*K2Lj*np.sqrt(bxi*bxj)*byi*byj*np.exp(1j*(muxi-muxj+2*muyi+2*muyj))
                            # --- geometric/chromatic
                            if 'h21001' in locals(): d21001 = -1j/16*K2Li*K2Lj*bxi*bxj**(3./2)*dxi*(np.exp(1j*muxj)-\
                                                                         2*np.exp(1j*(2*muxi-muxj))+\
                                                                           np.exp(-1j*(2*muxi-3*muxj)))
                            if 'h30001' in locals(): d30001 = -1j/16*K2Li*K2Lj*bxi*bxj**(3./2)*dxi*(np.exp(3j*muxj)-\
                                                                           np.exp(1j*(2*muxi+muxj)))
                            if 'h10021' in locals(): d10021 = 1j/16*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*(muxj-2*muyj))-\
                                                                               np.exp(1j*(2*muxi-muxj-2*muyj)))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj-2*muyi))-\
                                                                              np.exp(1j*(muxj-2*muyj)))
                            if 'h10111' in locals(): d10111 = 1j/8*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*muxj)-\
                                                                              np.exp(1j*(2*muxi-muxj)))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj-2*muyi+2*muyj))-\
                                                                              np.exp(1j*(muxj+2*muyi-2*muyj)))
                            if 'h10201' in locals(): d10201 = 1j/16*K2Li*K2Lj*bxi*np.sqrt(bxj)*byj*dxi*(np.exp(1j*(muxj+2*muyj))-\
                                                                               np.exp(1j*(2*muxi-muxj+2*muyj)))+\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxj)*byi*byj*dxi*(np.exp(1j*(muxj+2*muyi))-\
                                                                              np.exp(1j*(muxj+2*muyj)))
                            if 'h11002' in locals(): d11002 = 1j/4*K2Li*K2Lj*bxi*bxj*dxi*dxj*np.exp(2j*(muxi-muxj))+\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi**2*(np.exp(1j*(muxi-muxj))-\
                                                                                     np.exp(-1j*(muxi-muxj)))
                            if 'h20002' in locals(): d20002 = 1j/4*K2Li*K2Lj*bxi*bxj*dxi*dxj*np.exp(2j*muxi)+\
                                     1j/16*K2Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi**2*(np.exp(1j*(muxi+muxj))-\
                                                                                     np.exp(-1j*(muxi-3*muxj)))
                            if 'h00112' in locals(): d00112 = 1j/4*K2Li*K2Lj*byi*byj*dxi*dxj*np.exp(2j*(muyi-muyj))-\
                                     1j/8*K2Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi**2*(np.exp(1j*(muxi-muxj))-\
                                                                                 np.exp(-1j*(muxi-muxj)))
                            if 'h00202' in locals(): d00202 = 1j/4*K2Li*K2Lj*byi*byj*dxi*dxj*np.exp(2j*muyi)-\
                                     1j/16*K2Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi**2*(np.exp(1j*(muxi-muxj+2*muyj))-\
                                                                                  np.exp(-1j*(muxi-muxj-2*muyj)))
                            if 'h10003' in locals(): d10003 = 1j/4*K2Li*K2Lj*np.sqrt(bxi)*bxj*dxi**2*dxj*(np.exp(1j*muxi)-\
                                                                                 np.exp(-1j*(muxi-2*muxj)))
                            if 'h00004' in locals(): d00004 = 1j/4*K2Li*K2Lj*np.sqrt(bxi*bxj)*dxi**2*dxj**2*np.exp(1j*(muxi-muxj))

                            if j > i:
                                if 'h22000' in locals(): h22000 += d22000
                                if 'h31000' in locals(): h31000 += d31000
                                if 'h11110' in locals(): h11110 += d11110
                                if 'h11200' in locals(): h11200 += d11200
                                if 'h40000' in locals(): h40000 += d40000
                                if 'h20020' in locals(): h20020 += d20020
                                if 'h20110' in locals(): h20110 += d20110
                                if 'h20200' in locals(): h20200 += d20200
                                if 'h00220' in locals(): h00220 += d00220
                                if 'h00310' in locals(): h00310 += d00310
                                if 'h00400' in locals(): h00400 += d00400
                                if 'h21001' in locals(): h21001 += d21001
                                if 'h30001' in locals(): h30001 += d30001
                                if 'h10021' in locals(): h10021 += d10021
                                if 'h10111' in locals(): h10111 += d10111
                                if 'h10201' in locals(): h10201 += d10201
                                if 'h11002' in locals(): h11002 += d11002
                                if 'h20002' in locals(): h20002 += d20002
                                if 'h00112' in locals(): h00112 += d00112
                                if 'h00202' in locals(): h00202 += d00202
                                if 'h10003' in locals(): h10003 += d10003
                                if 'h00004' in locals(): h00004 += d00004

                            elif j < i:
                                if 'h22000' in locals(): h22000 -= d22000
                                if 'h31000' in locals(): h31000 -= d31000
                                if 'h11110' in locals(): h11110 -= d11110
                                if 'h11200' in locals(): h11200 -= d11200
                                if 'h40000' in locals(): h40000 -= d40000
                                if 'h20020' in locals(): h20020 -= d20020
                                if 'h20110' in locals(): h20110 -= d20110
                                if 'h20200' in locals(): h20200 -= d20200
                                if 'h00220' in locals(): h00220 -= d00220
                                if 'h00310' in locals(): h00310 -= d00310
                                if 'h00400' in locals(): h00400 -= d00400
                                if 'h21001' in locals(): h21001 -= d21001
                                if 'h30001' in locals(): h30001 -= d30001
                                if 'h10021' in locals(): h10021 -= d10021
                                if 'h10111' in locals(): h10111 -= d10111
                                if 'h10201' in locals(): h10201 -= d10201
                                if 'h11002' in locals(): h11002 -= d11002
                                if 'h20002' in locals(): h20002 -= d20002
                                if 'h00112' in locals(): h00112 -= d00112
                                if 'h00202' in locals(): h00202 -= d00202
                                if 'h10003' in locals(): h10003 -= d10003
                                if 'h00004' in locals(): h00004 -= d00004

                    if ej[0] == 1:
                        Lj,bnj,bxj,byj,dxj,muxj,muyj = ej[1],ej[2],ej[3],ej[4],ej[5],ej[6],ej[7]
                        K1Lj = Lj*bnj
                        if 'h21001' in locals(): d21001 = -1j/32*K2Li*K1Lj*bxi**(3./2)*bxj*(np.exp(1j*muxi)+\
                                                                        np.exp(1j*(3*muxi-2*muxj))-\
                                                                      2*np.exp(-1j*(muxi-2*muxj)))
                        if 'h30001' in locals(): d30001 = -1j/32*K2Li*K1Lj*bxi**(3./2)*bxj*(np.exp(3j*muxi)-\
                                                                        np.exp(1j*(muxi+2*muxj)))
                        if 'h10021' in locals(): d10021 = 1j/32*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*(muxi-2*muyi))-\
                                                                            np.exp(-1j*(muxi-2*muxj+2*muyi)))+\
                                      1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi-2*muyi))-\
                                                                            np.exp(1j*(muxi-2*muyj)))
                        if 'h10111' in locals(): d10111 = 1j/16*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*muxi)-\
                                                                            np.exp(-1j*(muxi-2*muxj)))+\
                                      1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi-2*muyi+2*muyj))-\
                                                                            np.exp(1j*(muxi+2*muyi-2*muyj)))
                        if 'h10201' in locals(): d10201 = 1j/32*K2Li*K1Lj*np.sqrt(bxi)*bxj*byi*(np.exp(1j*(muxi+2*muyi))-\
                                                                            np.exp(-1j*(muxi-2*muxj-2*muyi)))-\
                                      1j/16*K2Li*K1Lj*np.sqrt(bxi)*byi*byj*(np.exp(1j*(muxi+2*muyi))-\
                                                                            np.exp(1j*(muxi+2*muyj)))
                        if 'h11002' in locals(): d11002 = -1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*(muxi-muxj))+\
                                       1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(-2j*(muxi-muxj))
                        if 'h20002' in locals(): d20002 = -1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*muxi)+\
                                       1j/8*K2Li*K1Lj*bxi*bxj*dxi*np.exp(2j*muxj)
                        if 'h00112' in locals(): d00112 = -1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*(muyi-muyj))+\
                                       1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(-2j*(muyi-muyj))
                        if 'h00202' in locals(): d00202 = -1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*muyi)+\
                                       1j/8*K2Li*K1Lj*byi*byj*dxi*np.exp(2j*muyj)
                        if 'h10003' in locals(): d10003 =  1j/4*K2Li*K1Lj*bxi*np.sqrt(bxj)*dxi*dxj*(np.exp(1j*muxj)-\
                                                                                np.exp(1j*(2*muxi-muxj)))-\
                                       1j/8*K2Li*K1Lj*np.sqrt(bxi)*bxj*dxi**2*(np.exp(1j*muxi)-\
                                                                               np.exp(-1j*(muxi-2*muxj)))
                        if 'h00004' in locals(): d00004 = -1j/4*K2Li*K1Lj*np.sqrt(bxi*bxj)*dxi**2*dxj*np.exp(1j*(muxi-muxj))+\
                                  1j/4*K2Li*K1Lj*np.sqrt(bxi*bxj)*dxi**2*dxj*np.exp(-1j*(muxi-muxj))

                        if j > i:
                            if 'h21001' in locals(): h21001 += d21001
                            if 'h30001' in locals(): h30001 += d30001
                            if 'h10021' in locals(): h10021 += d10021
                            if 'h10111' in locals(): h10111 += d10111
                            if 'h10201' in locals(): h10201 += d10201
                            if 'h11002' in locals(): h11002 += d11002
                            if 'h20002' in locals(): h20002 += d20002
                            if 'h00112' in locals(): h00112 += d00112
                            if 'h00202' in locals(): h00202 += d00202
                            if 'h10003' in locals(): h10003 += d10003
                            if 'h00004' in locals(): h00004 += d00004

                        elif j < i:
                            if 'h21001' in locals(): h21001 -= d21001
                            if 'h30001' in locals(): h30001 -= d30001
                            if 'h10021' in locals(): h10021 -= d10021
                            if 'h10111' in locals(): h10111 -= d10111
                            if 'h10201' in locals(): h10201 -= d10201
                            if 'h11002' in locals(): h11002 -= d11002
                            if 'h20002' in locals(): h20002 -= d20002
                            if 'h00112' in locals(): h00112 -= d00112
                            if 'h00202' in locals(): h00202 -= d00202
                            if 'h10003' in locals(): h10003 -= d10003
                            if 'h00004' in locals(): h00004 -= d00004

            if ei[0] == 1:
                Li,bni,bxi,byi,dxi,muxi,muyi = ei[1],ei[2],ei[3],ei[4],ei[5],ei[6],ei[7]
                K1Li = Li*bni
                for j,ej in enumerate(self.h12table):
                    if ej[0] == 2:
                        Lj,bnj,bxj,byj,dxj,muxj,muyj = ej[1],ej[2],ej[3],ej[4],ej[5],ej[6],ej[7]
                        K2Lj = Lj*bnj
                        if 'h11002' in locals(): d11002 = -1j/8*K1Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi*(np.exp(1j*(muxi-muxj))-\
                                                                                    np.exp(-1j*(muxi-muxj)))
                        if 'h20002' in locals(): d20002 = -1j/16*K1Li*K2Lj*np.sqrt(bxi)*bxj**(3./2)*dxi*(np.exp(1j*(muxi+muxj))-\
                                                                                     np.exp(-1j*(muxi-3*muxj)))
                        if 'h00112' in locals(): d00112 = 1j/8*K1Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi*(np.exp(1j*(muxi-muxj))-\
                                                                               np.exp(-1j*(muxi-muxj)))
                        if 'h00202' in locals(): d00202 = 1j/16*K1Li*K2Lj*np.sqrt(bxi*bxj)*byj*dxi*(np.exp(1j*(muxi-muxj+2*muyj))-\
                                                                           np.exp(-1j*(muxi-muxj-2*muyj)))

                        if j > i:
                            if 'h11002' in locals(): h11002 += d11002
                            if 'h20002' in locals(): h20002 += d20002
                            if 'h00112' in locals(): h00112 += d00112
                            if 'h00202' in locals(): h00202 += d00202

                        elif j < i:
                            if 'h11002' in locals(): h11002 -= d11002
                            if 'h20002' in locals(): h20002 -= d20002
                            if 'h00112' in locals(): h00112 -= d00112
                            if 'h00202' in locals(): h00202 -= d00202

                    if ej[0] == 1:
                        if i != j:
                            Lj,bnj,bxj,byj,dxj,muxj,muyj = ej[1],ej[2],ej[3],ej[4],ej[5],ej[6],ej[7]
                            K1Lj = Lj*bnj
                            if 'h11002' in locals(): d11002 = 1j/16*K1Li*K1Lj*bxi*bxj*np.exp(2j*(muxi-muxj))
                            if 'h20002' in locals(): d20002 = 1j/16*K1Li*K1Lj*bxi*bxj*np.exp(2j*muxi)
                            if 'h00112' in locals(): d00112 = 1j/16*K1Li*K1Lj*byi*byj*np.exp(2j*(muyi-muyj))
                            if 'h00202' in locals(): d00202 = 1j/16*K1Li*K1Lj*byi*byj*np.exp(2j*muyi)
                            if 'h10003' in locals(): d10003 = 1j/8*K1Li*K1Lj*np.sqrt(bxi)*bxj*dxi*(np.exp(1j*muxi)-\
                                                                               np.exp(-1j*(muxi-2*muxj)))
                            if 'h00004' in locals(): d00004 = 1j/4*K1Li*K1Lj*np.sqrt(bxi*bxj)*dxi*dxj*np.exp(1j*(muxi-muxj))
                        if j > i:
                            if 'h11002' in locals(): h11002 += d11002
                            if 'h20002' in locals(): h20002 += d20002
                            if 'h00112' in locals(): h00112 += d00112
                            if 'h00202' in locals(): h00202 += d00202
                            if 'h10003' in locals(): h10003 += d10003
                            if 'h00004' in locals(): h00004 += d00004
                        if j < i:
                            if 'h11002' in locals(): h11002 -= d11002
                            if 'h20002' in locals(): h20002 -= d20002
                            if 'h00112' in locals(): h00112 -= d00112
                            if 'h00202' in locals(): h00202 -= d00202
                            if 'h10003' in locals(): h10003 -= d10003
                            if 'h00004' in locals(): h00004 -= d00004

        h2list = []
        if 'h22000' in locals(): h2list.append(('h22000',h22000*1j/64))
        if 'h31000' in locals(): h2list.append(('h31000',h31000*1j/32))
        if 'h11110' in locals(): h2list.append(('h11110',h11110*1j/16))
        if 'h11200' in locals(): h2list.append(('h11200',h11200*1j/32))
        if 'h40000' in locals(): h2list.append(('h40000',h40000*1j/64))
        if 'h20020' in locals(): h2list.append(('h20020',h20020*1j/64))
        if 'h20110' in locals(): h2list.append(('h20110',h20110*1j/32))
        if 'h20200' in locals(): h2list.append(('h20200',h20200*1j/64))
        if 'h00220' in locals(): h2list.append(('h00220',h00220*1j/64))
        if 'h00310' in locals(): h2list.append(('h00310',h00310*1j/32))
        if 'h00400' in locals(): h2list.append(('h00400',h00400*1j/64))
        if 'h21001' in locals(): h2list.append(('h21001',h21001     ))
        if 'h30001' in locals(): h2list.append(('h30001',h30001     ))
        if 'h10021' in locals(): h2list.append(('h10021',h10021     ))
        if 'h10111' in locals(): h2list.append(('h10111',h10111     ))
        if 'h10201' in locals(): h2list.append(('h10201',h10201     ))
        if 'h11002' in locals(): h2list.append(('h11002',h11002     ))
        if 'h20002' in locals(): h2list.append(('h20002',h20002     ))
        if 'h00112' in locals(): h2list.append(('h00112',h00112     ))
        if 'h00202' in locals(): h2list.append(('h00202',h00202     ))
        if 'h10003' in locals(): h2list.append(('h10003',h10003     ))
        if 'h00004' in locals(): h2list.append(('h00004',h00004     ))

        self.h2 = dict(h2list)

    def geth1_old(self):
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
                #print K2Li,bxi,byi,dxi,muxi,muyi
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
                #print K1Li,bxi,byi,dxi,muxi,muyi
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)

            '''

            if ei.__class__.__name__ == 'bend':
                #entrance fringe
                K1Li = -np.tan(ei.e1)/ei.R
                bxi,byi,dxi,muxi,muyi = self.twx[0,i],self.twy[0,i],self.etax[i],self.mux[i],self.muy[i]
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
                bxi,byi,dxi,muxi,muyi = self.twx[0,i+1],self.twy[0,i+1],self.etax[i+1],self.mux[i+1],self.muy[i+1]
                h11001 += K1Li*bxi
                h00111 -= K1Li*byi
                h20001 += K1Li*bxi*np.exp(2j*muxi)
                h00201 -= K1Li*byi*np.exp(2j*muyi)
                h10002 += K1Li*dxi*np.sqrt(bxi)*np.exp(1j*muxi)
            '''

        self.h1 = {'h11001':h11001/4,'h00111':h00111/4,'h20001':h20001/8,'h00201':h00201/8,
                   'h10002':h10002/2,'h21000':h21000/8,'h30000':h30000/24,'h10110':h10110/4,
                   'h10020':h10020/8,'h10200':h10200/8}

    def geth2_old(self):
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

        self.h2 = {'h22000':h22000*1j/64,'h31000':h31000*1j/32,\
                   'h11110':h11110*1j/16,'h11200':h11200*1j/32,\
                   'h40000':h40000*1j/64,
                   'h20020':h20020*1j/64,'h20110':h20110*1j/32,\
                   'h20200':h20200*1j/64,'h00220':h00220*1j/64,\
                   'h00310':h00310*1j/32,
                   'h00400':h00400*1j/64,'h21001':h21001,\
                   'h30001':h30001,'h10021':h10021,\
                   'h10111':h10111,'h10201':h10201,
                   'h11002':h11002,'h20002':h20002,\
                   'h00112':h00112,'h00202':h00202,\
                   'h10003':h10003,'h00004':h00004}


    def getCoupleOrm(self,cindex=None,bindex=None):
        '''
        Matrix-based ORM
        '''
        if not cindex:
            cindex = self.getIndex('kick')
        if not bindex:
            bindex = self.getIndex('moni')
        nbpm,ncor = len(bindex),len(cindex)
        rm = np.zeros((2*nbpm,2*ncor))
        b = np.array([0.,1.,0.,0.])
        for ic,ci in enumerate(cindex):
            m = self.getOneTurnMatrix(ci)[:4,:4]
            a = np.eye(4)-m
            X = np.linalg.solve(a,b)
            X = np.reshape(X,(4,1))
            for ib,bi in enumerate(bindex):
                mij = self.getTransMat(ci,bi)[:4,:4]
                rme = mij*X
                rm[ib,ic] = rme[0]
                rm[nbpm+ib,ic] = rme[2]
                sys.stdout.flush()
                sys.stdout.write('\rH: cor index: %04i, bpm index: %04i'
                                 %(ic+1,ib+1))
        b = np.array([0.,0.,0.,1.])
        for ic,ci in enumerate(cindex):
            m = self.getOneTurnMatrix(ci)[:4,:4]
            a = np.eye(4)-m
            X = np.linalg.solve(a,b)
            X = np.reshape(X,(4,1))
            for ib,bi in enumerate(bindex):
                mij = self.getTransMat(ci,bi)[:4,:4]
                rme = mij*X
                rm[ib,ic+ncor] = rme[0]
                rm[nbpm+ib,ic+ncor] = rme[2]
                sys.stdout.flush()
                sys.stdout.write('\rV: cor index: %04i, bpm index: %04i'
                                 %(ic+1,ib+1))
        self.coupleorm = rm


    def getOrm(self,bpms=None,cors=None):
        '''
        get Obit Response Matrix to correctors from optics
        so far, no coupling included
        '''
        if not bpms:
            bpms = self.getElements('moni')
        if not cors:
            cors = self.getElements('kick')
        ormx = np.zeros((len(bpms),len(cors)))
        ormy = np.zeros((len(bpms),len(cors)))
        ic = [self.bl.index(cor) for cor in cors]
        ib = [self.bl.index(bpm) for bpm in bpms]
        for m,cor in enumerate(cors):
            for n,bpm in enumerate(bpms):
                sbx = np.sqrt(self.betax[ic[m]]*self.betax[ib[n]])
                sby = np.sqrt(self.betay[ic[m]]*self.betay[ib[n]])
                cpx = np.cos(abs(self.mux[ic[m]]-self.mux[ib[n]])*twopi - \
                             np.pi*self.nux)
                cpy = np.cos(abs(self.muy[ic[m]]-self.muy[ib[n]])*twopi - \
                             np.pi*self.nuy)
                ormx[n,m] = sbx*cpx
                ormy[n,m] = sby*cpy
        ormx /= 2*np.sin(np.pi*self.nux)
        ormy /= 2*np.sin(np.pi*self.nuy)
        self.ormx,self.ormy = ormx,ormy


    def getRespMat(self,bpms=None,quads=None,dk=0.001,
                   verbose=False):
        '''
        get response matrices for
        1. phase advance (parm)
        2. tune (tunerm)
        3. horizontal dispersion (disprm)
        4. beta-function (betarm)
        depending on given quadrupoles
        '''
        nux0 = self.nux
        nuy0 = self.nuy
        if not bpms:
            bpms = self.getElements('moni')
        bpmIndex = [self.bl.index(bpm) for bpm in bpms]
        bx0 = self.betax[bpmIndex]
        by0 = self.betay[bpmIndex]
        phx0 = self.mux[bpmIndex]
        phy0 = self.muy[bpmIndex]
        etax0 = self.etax[bpmIndex]
        if not quads:
            quads = self.getElements('quad',prefix='q')
        nbpm,nquad = len(bpms),len(quads)

        betarmx = np.zeros((nbpm,nquad))
        betarmy = np.zeros((nbpm,nquad))
        parmx = np.zeros((nbpm,nquad))
        parmy = np.zeros((nbpm,nquad))
        disprm = np.zeros((nbpm,nquad))
        tunerm = np.zeros((2,nquad))

        for i,quad in enumerate(quads):
            quad.put('K1',quad.K1+dk)
            self.update()
            dbx = self.betax[bpmIndex]-bx0
            dby = self.betay[bpmIndex]-by0
            dphx = self.mux[bpmIndex]-phx0
            dphy = self.muy[bpmIndex]-phy0
            detax = self.etax[bpmIndex]-etax0
            dnux = self.nux-nux0
            dnuy = self.nuy-nuy0
            betarmx[:,i] = dbx/dk
            betarmy[:,i] = dby/dk
            parmx[:,i] = dphx/dk
            parmy[:,i] = dphy/dk
            disprm[:,i] = detax/dk
            tunerm[:,i] = np.array([dnux/dk,dnuy/dk])
            quad.put('K1',quad.K1-dk)
            if verbose:
                sys.stdout.write('\r --- %04i out of %04i done (%3i%%) ---'\
                                 %(i+1,nquad,(i+1)*100./nquad))
                sys.stdout.flush()
        self.update()
        self.betarmx = betarmx
        self.betarmy = betarmy
        self.parmx = parmx
        self.parmy = parmy
        self.disprm = disprm
        self.tunerm = tunerm


    def getChrm(self,sexts=None,dk=0.01,verbose=False):
        '''
        get chromaticity response matrix
        '''
        if not hasattr(self, 'chx'):
            self.chrom()
        chx0,chy0 = self.chx,self.chy
        if not sexts:
            sexts = self.getElements('sext',prefix='sm')
        nsext = len(sexts)
        chrm = np.zeros((2,nsext))
        for i,sext in enumerate(sexts):
            sext.put('K2',sext.K2+dk)
            self.chrom()
            chrm[:,i] = np.array([(self.chx-chx0)/dk,(self.chy-chy0)/dk])
            sext.put('K2',sext.K2-dk)
            if verbose:
                sys.stdout.write('\r%04i out of %04i: %s'%(i+1,nsext,sext.name))
                sys.stdout.flush()
        self.chrom()
        self.chrm = chrm


    def finddyap(self,xmin=-3e-2, xmax=3e-2, ymin=0, ymax=1.5e-2,
                 nx=61, ny=16, dp=0, nturn=512):
        '''
        find dynamic aperture with matrix tracking and thin-len kicks
        not very accurate for off-momentum dyap
        '''
        x = np.linspace(xmin,xmax,nx)
        y = np.linspace(ymin,ymax,ny)
        xgrid,ygrid = np.meshgrid(x,y)
        xin = np.zeros((7,nx*ny))
        xin[-1] = np.arange(nx*ny)
        xin[0] = xgrid.flatten()
        xin[2] = ygrid.flatten()
        xin[5] = dp
        xout = self.matrack(xin,nturn=nturn,checkap=True)
        dyap = xout[6]
        dyap = dyap.reshape(xgrid.shape)
        self.dyap = {'xgrid':xgrid,'ygrid':ygrid,'dyap':dyap,'dp':dp}


# --- symplectic tracking based simulation
    def finddyapsym4(self,xmin=-3e-2,xmax=3e-2,ymin=0,ymax=1.5e-2,
                     nx=61,ny=16,dp=0,nturn=512,dfu=True,naf=True,
                     tunewindow=0,verbose=0):
        '''
        find dynamic aperture symplectic kick-drift
        '''
        x = np.linspace(xmin,xmax,int(nx))
        y = np.linspace(ymin,ymax,int(ny))
        xgrid,ygrid = np.meshgrid(x,y)
        xin = np.zeros((7,nx*ny))
        xin[-1] = np.arange(nx*ny)
        xin[0] = xgrid.flatten()
        xin[2] = ygrid.flatten()
        xin[5] = dp
        tbt = np.zeros((nturn,6,nx*ny))
        for i in xrange(nturn):
            xin[:6] = self.eletrack(xin[:6])[-1]
            tbt[i] = np.array(xin[:6])
            if verbose:
                sys.stdout.write('\r--- tracking: %04i out of %04i is being done (%3i%%) ---'%
                                 (i+1,nturn,(i+1)*100./nturn))
                sys.stdout.flush()
        xin[6] = chkap(xin)
        dyap = xin[6]
        dyap = dyap.reshape(xgrid.shape)
        self.dyap = {'xgrid':xgrid,'ygrid':ygrid,'dyap':dyap,'dp':dp,'tbt':tbt}
        # --- if diffusion
        if dfu:
            if tunewindow:
                rngx = [self.nux-int(self.nux)-tunewindow,self.nux-int(self.nux)+tunewindow]
                rngy = [self.nuy-int(self.nuy)-tunewindow,self.nuy-int(self.nuy)+tunewindow]
            else:
                rngx,rngy = [0.,.5],[0.,.5]
            fmx,fmy = np.zeros(nx*ny),np.zeros(nx*ny)
            df1 = np.zeros(nx*ny)
            for i in xrange(nx*ny):
                if verbose:
                    sys.stdout.write('\r--- diffusion: %04i out of %04i is being done (%3i%%) ---'%
                                     (i+1,nx*ny,(i+1)*100./nx/ny))
                    sys.stdout.flush()
                if any(np.isnan(tbt[:,0,i])) or any(np.isnan(tbt[:,2,i])):
                    df1[i] = np.nan
                    continue
                else:
                    if naf:
                        v1x = naff(tbt[:nturn/2,0,i],ni=1,rng=rngx,verbose=0)[0][0]
                        v2x = naff(tbt[nturn/2:,0,i],ni=1,rng=rngx,verbose=0)[0][0]
                        v1y = naff(tbt[:nturn/2,2,i],ni=1,rng=rngy,verbose=0)[0][0]
                        v2y = naff(tbt[nturn/2:,2,i],ni=1,rng=rngy,verbose=0)[0][0]
                    else:
                        v1x = getTuneTbT(tbt[:nturn/2,0,i],rng=rngx,verbose=0,hann=1)[0]
                        v2x = getTuneTbT(tbt[nturn/2:,0,i],rng=rngx,verbose=0,hann=1)[0]
                        v1y = getTuneTbT(tbt[:nturn/2,2,i],rng=rngy,verbose=0,hann=1)[0]
                        v2y = getTuneTbT(tbt[nturn/2:,2,i],rng=rngy,verbose=0,hann=1)[0]
                    #print '%15.6e%15.6e%15.6e%15.6e'%(v1x,v2x,v1y,v2y)
                    dvx2 = (v1x-v2x)**2
                    dvy2 = (v1y-v2y)**2
                    df1[i] = np.log10(np.sqrt(dvx2+dvy2))
                    fmx[i] = (v1x+v2x)/2
                    fmy[i] = (v1y+v2y)/2
            df = df1.reshape((ny,nx))
            self.dyap['dfu'] = df
            self.dyap['fma'] = np.array([fmx,fmy,df1]).transpose()

    def pltdyap(self,xyunit='mm',figsize=(8,8),mode='contourf',fn=None,fndfu=None):
        '''
        plot dynamic aperture
        '''
        if self.dyap:
            unit = 1
            if xyunit == 'mm':
                unit = 1e3
            elif xyunit == 'cm':
                unit = 1e2
            else:
                print("undefined unit: '%d'"%xyunit)
            plt.figure(figsize=figsize)
            plt.contourf(self.dyap['xgrid']*unit,self.dyap['ygrid']*unit,
                         self.dyap['dyap'],1)
            plt.xlabel('x[%s]'%xyunit)
            plt.ylabel('y[%s]'%xyunit)
            plt.title('Dynamic aperture for dp/p = %.2f'%self.dyap['dp'])
            if fn:
                plt.savefig(fn)
                plt.close()
            else:
                plt.show()
            if 'dfu' in self.dyap.keys():
                plt.figure(figsize=figsize)
                maskeddfu = np.ma.array(self.dyap['dfu'],
                                        mask=np.isnan(self.dyap['dfu']))
                if mode == 'contourf':
                    df = plt.contourf(self.dyap['xgrid']*unit,
                                      self.dyap['ygrid']*unit,
                                      maskeddfu)
                else:
                    df = plt.pcolor(self.dyap['xgrid']*unit,
                                    self.dyap['ygrid']*unit,
                                    maskeddfu)
                plt.xlabel('x [%s]'%xyunit)
                plt.ylabel('y [%s]'%xyunit)
                plt.title('Dynamic aperture for dp/p = %.2f'%self.dyap['dp'])
                cb = plt.colorbar(df)
                cb.ax.set_ylabel(r'$log_{10}\sqrt{\Delta\nu_x^2+\Delta\nu_y^2}$')
                if fndfu:
                    plt.savefig(fndfu)
                    plt.close()
                else:
                    plt.show()
        else:
            print('dynamic aperture has been not calculated yet.\n')

    def pltFma(self,xmin=0,xmax=0.5,ymin=0,ymax=0.5,order=5,
               crossHalfInt=False,ms=2,figsize=(5,5)):
        '''
        plot FMA
        '''
        if crossHalfInt:
            cs = CSNormalization(self.betax[0],self.alfax[0],
                                 self.betay[0],self.alfay[0],combine=True)
            '''
            sqtbx = np.sqrt(self.betax[0])
            ax = self.alfax[0]
            sqtby = np.sqrt(self.betay[0])
            ay = self.alfay[0]
            m = np.mat([[1/sqtbx,0,0,0],[ax/sqtbx,sqtbx,0,0],
                        [0,0,1/sqtby,0],[0,0,ay/sqtby,sqtby]])
            cs = np.mat([[1,-1j,0,0],[0,0,1,-1j]])
            '''
            n = self.dyap['tbt'].shape[2]
            for i in xrange(n):
                a0 = np.array(cs*self.dyap['tbt'][:,:4,i].transpose())
                xpw0 = np.fft.fft(a0[0])
                self.dyap['fma'][i,0] = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                xpw0 = np.fft.fft(a0[1])
                self.dyap['fma'][i,1] = pickPeak(np.abs(xpw0),rng=[0,1])[0]

        plt.figure(figsize=figsize)
        plt.scatter(self.dyap['fma'][:,0],self.dyap['fma'][:,1],s=ms,
                    c=self.dyap['fma'][:,2],alpha=1,edgecolor='none')
        color = ['','k','r','b','y','g','m','c','c','c','c']
        for nx in range(-1*order,order):
            for ny in range(-1*order,order):
                for p in range(nx-order,nx+order):
                    if nx==0 and ny==0:
                        continue
                    if ny != 0:
                        xmin1,xmax1 = xmin,xmax
                        ymin1 = (p-nx*xmin1)*1./ny
                        ymax1 = (p-nx*xmax1)*1./ny
                    else:
                        xmin1 = p*1./nx
                        xmax1 = p*1./nx
                        ymin1,ymax1 = ymin,ymax
                    if abs(nx)+abs(ny) <= abs(order):
                        plt.plot([xmin1,xmax1],[ymin1,ymax1],'-',color=color[abs(nx)+abs(ny)],lw=0.5)
        plt.axis([xmin,xmax,ymin,ymax])
        plt.show()

    def pltPhaseSpace(self,figsize=(16,6),densityGap=10,fn=None):
        if not hasattr(self,'dyap'):
            print('no turb-by-turn data available')
        else:
            plt.figure(figsize=figsize)
            n1,n2,n3 = self.dyap['tbt'].shape
            for i in xrange(0,n3,densityGap):
                if any(np.isnan(self.dyap['tbt'][:,0,i])) or \
                   any(np.isnan(self.dyap['tbt'][:,2,i])):
                    continue
                else:
                    plt.subplot(121)
                    plt.plot(self.dyap['tbt'][:,0,i],self.dyap['tbt'][:,1,i],'b.')
                    plt.xlabel('x(m)')
                    plt.ylabel("x'(rad)")
                    plt.subplot(122)
                    plt.plot(self.dyap['tbt'][:,2,i],self.dyap['tbt'][:,3,i],'r.')
                    plt.xlabel('y(m)')
                    plt.ylabel("y'(rad)")
            if fn:
                plt.savefig(fn)
                plt.close()
            else:
                plt.show()



    def tuneShiftWithAmpTrack(self,Amin=-2e-2,Amax=2e-2,nA=40,
                              nturn=128,plane='x',polyorder=4,
                              sA=1e-8,verbose=False,tunewindow=0,
                              crossHalfInt=0):
        '''
        get tune-shift-with-amplitude by tracking
        '''
        dA = np.linspace(Amin,Amax,int(nA))
        xin = np.zeros((6,nA))
        if plane == 'x':
            xin[0] = dA
            xin[2] = nA*[sA]
        elif plane == 'y':
            xin[2] = dA
            xin[0] = nA*[sA]
        tbt = np.zeros((nturn,6,nA))
        for i in xrange(nturn):
            xin[:6] = self.eletrack(xin[:6])[-1]
            tbt[i] = np.array(xin[:6])
            sys.stdout.write(
                '\r--- tracking: %04i out of %04i is being done (%3i%%) ---'
                %(i+1,nturn,(i+1)*100./nturn))
            sys.stdout.flush()
        nu = []
        if tunewindow:
            rngx = [self.nux-int(self.nux)-tunewindow,self.nux-int(self.nux)+tunewindow]
            rngy = [self.nuy-int(self.nuy)-tunewindow,self.nuy-int(self.nuy)+tunewindow]
        else:
            rngx,rngy = [0.,.5],[0.,.5]
        if crossHalfInt:
            csm = CSNormalization(self.betax[0],self.alfax[0],
                                  self.betay[0],self.alfay[0],combine=True)
            '''
            sqtbx = np.sqrt(self.betax[0])
            ax = self.alfax[0]
            sqtby = np.sqrt(self.betay[0])
            ay = self.alfay[0]
            m = np.mat([[1/sqtbx,0,0,0],[ax/sqtbx,sqtbx,0,0],
                        [0,0,1/sqtby,0],[0,0,ay/sqtby,sqtby]])
            cs = np.mat([[1,-1j,0,0],[0,0,1,-1j]])
            csm = cs*m
            '''
        for i in xrange(nA):
            if crossHalfInt:
                a0 = np.array(csm*tbt[:,:4,i].transpose())
                xpw0 = np.fft.fft(a0[0])
                nu1 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                xpw0 = np.fft.fft(a0[1])
                nu2 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                nu.append([dA[i],nu1,nu2])
            else:
                nu.append([dA[i],naff(tbt[:,0,i],ni=1,verbose=0,rng=rngx)[0][0],
                           naff(tbt[:,2,i],ni=1,verbose=0,rng=rngy)[0][0]])
        nu = np.array(nu)
        Anux = np.polyfit(nu[:,0],nu[:,1],polyorder)
        Anuy = np.polyfit(nu[:,0],nu[:,2],polyorder)
        if verbose:
            nuxfit = np.polyval(Anux,nu[:,0])
            nuyfit = np.polyval(Anuy,nu[:,0])
            plt.figure(figsize=(8,6))
            plt.plot(nu[:,0],nuxfit,'b-',
                     label=r"$\frac{d\nu_x}{d%s^2}=%.2f$"%(plane,Anux[-3]),lw=2)
            plt.plot(nu[:,0],nu[:,1],'bo')
            plt.xlabel(r'%s (m)'%plane,fontsize=18)
            plt.ylabel(r'$\nu_x$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            plt.show()
            plt.figure(figsize=(8,6))
            plt.plot(nu[:,0],nuyfit,'r-',
                     label=r"$\frac{d\nu_y}{d%s^2}=%.2f$"%(plane,Anuy[-3]),lw=2)
            plt.plot(nu[:,0],nu[:,2],'ro')
            plt.xlabel(r'%s (m)'%plane,fontsize=18)
            plt.ylabel(r'$\nu_y$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            plt.show()
        self.tuneamp = {'tbt':tbt,'dnuxdA':Anux,'dnuydA':Anuy,'plane':plane}

    def tswa(self,xyaxis=[1e-8,2.5e-2,1e-8,1e-2],nA=40,
             nturn=128,polyorder=4,verbose=False,tunewindow=0,
             crossHalfInt=0):
        '''
        get tune-shift-with-amplitude by tracking
        '''
        dAx = np.linspace(xyaxis[0],xyaxis[1],int(nA))
        dAy = np.linspace(xyaxis[2],xyaxis[3],int(nA))
        xin = np.zeros((6,nA))
        xin[0] = dAx
        xin[2] = dAy
        tbt = np.zeros((nturn,6,nA))
        for i in xrange(nturn):
            xin[:6] = self.eletrack(xin[:6])[-1]
            tbt[i] = np.array(xin[:6])
            sys.stdout.write(
                '\r--- tracking: %04i out of %04i is being done (%3i%%) ---'
                %(i+1,nturn,(i+1)*100./nturn))
            sys.stdout.flush()
        nu = []
        if tunewindow:
            rngx = [self.nux-int(self.nux)-tunewindow,self.nux-int(self.nux)+tunewindow]
            rngy = [self.nuy-int(self.nuy)-tunewindow,self.nuy-int(self.nuy)+tunewindow]
        else:
            rngx,rngy = [0.,.5],[0.,.5]
        if crossHalfInt:
            csm = CSNormalization(self.betax[0],self.alfax[0],
                                  self.betay[0],self.alfay[0],combine=True)
            '''
            sqtbx = np.sqrt(self.betax[0])
            ax = self.alfax[0]
            sqtby = np.sqrt(self.betay[0])
            ay = self.alfay[0]
            m = np.mat([[1/sqtbx,0,0,0],[ax/sqtbx,sqtbx,0,0],
                        [0,0,1/sqtby,0],[0,0,ay/sqtby,sqtby]])
            cs = np.mat([[1,-1j,0,0],[0,0,1,-1j]])
            csm = cs*m
            '''
        for i in xrange(nA):
            if crossHalfInt:
                a0 = np.array(csm*tbt[:,:4,i].transpose())
                xpw0 = np.fft.fft(a0[0])
                nu1 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                xpw0 = np.fft.fft(a0[1])
                nu2 = pickPeak(np.abs(xpw0),rng=[0,1])[0]
                nu.append([dAx[i],dAy[i],nu1,nu2])
            else:
                nu.append([dAx[i],dAy[i],
                           naff(tbt[:,0,i],ni=1,verbose=0,rng=rngx)[0][0],
                           naff(tbt[:,2,i],ni=1,verbose=0,rng=rngy)[0][0]])
        nu = np.array(nu)
        #Anux = np.polyfit(nu[:,0],nu[:,1],polyorder)
        #Anuy = np.polyfit(nu[:,0],nu[:,2],polyorder)
        if verbose:
            #nuxfit = np.polyval(Anux,nu[:,0])
            #nuyfit = np.polyval(Anuy,nu[:,0])
            plt.figure(figsize=(8,6))
            #plt.plot(nu[:,0],nuxfit,'b-',
            #         label=r"$\frac{d\nu_x}{d%s^2}=%.2f$"%(plane,Anux[-3]),lw=2)
            plt.plot(nu[:,0],nu[:,2],'r-o')
            plt.plot(nu[:,0],nu[:,3],'b-o')
            plt.xlabel('x (m)',fontsize=18)
            plt.ylabel(r'$\nu_{x,y}$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            plt.show()
            plt.figure(figsize=(8,6))
            #plt.plot(nu[:,0],nuyfit,'r-',
            #         label=r"$\frac{d\nu_y}{d%s^2}=%.2f$"%(plane,Anuy[-3]),lw=2)
            plt.plot(nu[:,1],nu[:,2],'r-o')
            plt.plot(nu[:,1],nu[:,3],'b-o')
            plt.xlabel('y (m)',fontsize=18)
            plt.ylabel(r'$\nu_{x,y}$',fontsize=18)
            plt.legend(loc='best',fontsize=18)
            plt.show()
        self.tuneamp = {'tbt':tbt,'nux_amp':nu}


    def finddyapXD(self,xmin=-3e-2,xmax=3e-2,dmin=-3e-2,dmax=3e-2,
                   nx=30,nd=30,nturn=256,dfu=True,naf=True,sA=1e-8,
                   tunewindow=0,verbose=0):
        '''
        find dynamic aperture x vs. delta with symplectic kick-drift
        sA: small amplitude to avoid zero amplitude
        '''
        x = np.linspace(xmin,xmax,int(nx))
        y = np.linspace(dmin,dmax,int(nd))
        xgrid,ygrid = np.meshgrid(x,y)
        xin = np.zeros((7,nx*nd))
        xin[-1] = np.arange(nx*nd)
        xin[0] = xgrid.flatten()
        xin[2] = np.ones(nx*nd)*sA
        xin[5] = ygrid.flatten()
        tbt = np.zeros((nturn,6,nx*nd))
        for i in xrange(nturn):
            xin[:6] = self.eletrack(xin[:6])[-1]
            tbt[i] = np.array(xin[:6])
            if verbose:
                sys.stdout.write('\r--- tracking: %04i out of %04i is being done (%3i%%) ---'%
                                 (i+1,nturn,(i+1)*100./nturn))
                sys.stdout.flush()
        xin[6] = chkap(xin)
        dyap = xin[6]
        dyap = dyap.reshape(xgrid.shape)
        self.dyapXD = {'xgrid':xgrid,'dgrid':ygrid,'dyap':dyap,'tbt':tbt}
        # --- if diffusion
        if dfu:
            df = np.zeros(nx*nd)
            if tunewindow:
                rngx = [self.nux-int(self.nux)-tunewindow,self.nux-int(self.nux)+tunewindow]
                rngy = [self.nuy-int(self.nuy)-tunewindow,self.nuy-int(self.nuy)+tunewindow]
            else:
                rngx,rngy = [0.,.5],[0.,.5]
            for i in xrange(nx*nd):
                if verbose:
                    sys.stdout.write('\r--- diffusion: %04i out of %04i is being done (%3i%%) ---'%
                                     (i+1,nx*nd,(i+1)*100./nx/nd))
                    sys.stdout.flush()
                if any(np.isnan(tbt[:,0,i])) or any(np.isnan(tbt[:,2,i])):
                    df[i] = np.nan
                    continue
                else:
                    if naf:
                        v1x = naff(tbt[:nturn/2,0,i],ni=1,rng=rngx,verbose=0)[0][0]
                        v2x = naff(tbt[nturn/2:,0,i],ni=1,rng=rngx,verbose=0)[0][0]
                        v1y = naff(tbt[:nturn/2,2,i],ni=1,rng=rngy,verbose=0)[0][0]
                        v2y = naff(tbt[nturn/2:,2,i],ni=1,rng=rngy,verbose=0)[0][0]
                    else:
                        v1x = getTuneTbT(tbt[:nturn/2,0,i],rng=rngx,verbose=0,hann=1)[0]
                        v2x = getTuneTbT(tbt[nturn/2:,0,i],rng=rngx,verbose=0,hann=1)[0]
                        v1y = getTuneTbT(tbt[:nturn/2,2,i],rng=rngy,verbose=0,hann=1)[0]
                        v2y = getTuneTbT(tbt[nturn/2:,2,i],rng=rngy,verbose=0,hann=1)[0]
                    dvx2 = (v1x-v2x)**2
                    dvy2 = (v1y-v2y)**2
                    df[i] = np.log10(np.sqrt(dvx2+dvy2))
            df = df.reshape(xgrid.shape)
            self.dyapXD['dfu'] = df

    def pltdyapXD(self,xyunit='mm',figsize=(8,6),mode='contourf',
                  fn=None,fndf=None):
        '''
        plot dynamic aperture
        '''
        if self.dyapXD:
            unit = 1
            if xyunit == 'mm':
                unit = 1e3
            elif xyunit == 'cm':
                unit = 1e2
            else:
                print("undefined unit: '%d'"%xyunit)
            plt.figure(figsize=figsize)
            plt.contourf(self.dyapXD['dgrid'],self.dyapXD['xgrid']*unit,
                         self.dyapXD['dyap'],1)
            plt.ylabel('x[%s]'%xyunit)
            plt.xlabel(r'$\delta=\frac{\Delta P}{P}$')
            plt.title('Dynamic aperture for x vs. delta')
            if fn:
                plt.savefig(fn)
                plt.close()
            else:
                plt.show()
            if 'dfu' in self.dyapXD.keys():
                plt.figure(figsize=figsize)
                maskeddfu = np.ma.array(self.dyapXD['dfu'],
                                        mask=np.isnan(self.dyapXD['dfu']))
                if mode == 'contourf':
                    df = plt.contourf(self.dyapXD['dgrid'],
                                      self.dyapXD['xgrid']*unit,
                                      maskeddfu,1)
                else:
                    df = plt.pcolor(self.dyapXD['dgrid'],
                                    self.dyapXD['xgrid']*unit,
                                    maskeddfu)
                plt.ylabel('x [%s]'%xyunit)
                plt.xlabel(r'$\delta=\frac{\Delta P}{P}$')
                plt.title('Dynamic aperture for x vs. delta')
                cb = plt.colorbar(df)
                cb.ax.set_ylabel(r'$log_{10}\sqrt{\Delta\nu_x^2+\Delta\nu_y^2}$')
                if fndf:
                    plt.savefig(fndf)
                    plt.close()
                else:
                    plt.show()
        else:
            print('dynamic aperture has been not calculated yet.\n')

# --- end of symplectic tracking based simulation


    def switchoffID(self,ids=None):
        '''
        switch off insertion devices,
        change it into drift
        '''
        print('\n === warning: switched-off IDs can NOT be swithed on ===\n')
        if not ids:
            ids = self.getElements('kmap') + self.getElements('wigg')
        for ie in ids:
            ii = self.bl.index(ie)
            replacement = drif(ie.name,L=ie.L)
            self.bl[ii] = replacement
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
            dk += v[m,:]/s[m]*np.dot(x, u[:,m])
        else:
            break
    xr = np.mat(x).reshape(-1,1)+np.mat(rm)*np.mat(dk).reshape(-1,1)
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

def rotmat(angle=np.pi/4):
    '''
    rotating matrix with a given angle
    facing the beam, rotate counter-clockwise is defined as positive
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    m = np.eye(6)
    for i in range(4):
        m[i,i] = c
    m[0,2] = s
    m[1,3] = s
    m[2,0] = -s
    m[3,1] = -s
    return m

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

def twmat(elem,s):
    '''
    Computes the transport and twiss matrix for transport
    through distance s in element elem.
    usage: mat = twmat(elem,s)
    '''
    if elem.__class__.__name__=='drif' or \
       elem.__class__.__name__=='sext' or \
       elem.__class__.__name__=='kick' or \
       elem.__class__.__name__=='rfca' or \
       elem.__class__.__name__=='aper' or \
       elem.__class__.__name__=='mult':
        fe = drif(L=s)
    elif elem.__class__.__name__ == 'quad':
        fe = quad(L=s,K1=elem.K1,tilt=elem.tilt)
    elif elem.__class__.__name__ == 'skew':
        fe = skew(L=s,K1=elem.K1)
    elif elem.__class__.__name__ == 'sole':
        fe = sole(L=s,KS=elem.KS)
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
        raise RuntimeError('unknown type for element "%s"'%elem.name)
    mxy = np.take(np.take(fe.tm,[0,1,2,3,5],axis=0),[0,1,2,3,5],axis=1)
    #my = np.take(np.take(fe.tm,[2,3,5],axis=0),[2,3,5],axis=1)
    return mxy,fe.tx,fe.ty

def twisstrans(twissmatrix,twiss0):
    '''
    get tiwss functions by twiss transport matrix
    '''
    return twissmatrix.dot(twiss0)

def phasetrans(m,tw0,neglen=False):
    '''
    get phase advance by transport matrix
    '''
    dph = np.arctan(m[0,1]/(m[0,0]*tw0[0]-m[0,1]*tw0[1]))/twopi
    if neglen:
        if dph>0:
            return dph-0.5
        else:
            return dph
    else:
        if dph<0:
            return dph+0.5
        else:
            return dph

def trans2twiss(a):
    '''
    calculate twiss matrix from transport matrix
    input (a) is a 2x2 coordinate transport matrix
    return a 3x3 twiss transport matrix [beta, alfa, gama]
    '''
    return np.array([[a[0,0]**2,-2*a[0,0]*a[0,1],a[0,1]**2],
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
        raise RuntimeError('plane must be \'x\' or \'y\'')
    if abs(r[0,0]+r[1,1]) >= 2:
        raise RuntimeError('det(M) >= 2: no stable solution in %s plane'%plane)
    mu = np.arccos((r[0,0]+r[1,1])/2)/twopi
    if r[0,1] < 0:
        mu = 1-mu
    s = np.sin(twopi*mu)
    beta = r[0,1]/s
    alfa = (r[0,0]-r[1,1])/(2*s)
    gama = -r[1,0]/s
    return mu,beta,alfa,gama


def matrix2disp(R):
    '''
    calculate dispersions based on transport matrix
    '''
    a = R[:4,:4]-np.eye(4)
    b = -R[:4,5]
    dxy = np.linalg.solve(a,b)
    return dxy[0],dxy[1],dxy[2],dxy[3]


def txt2latt(fn,verbose=False):
    '''
    read nls2-ii control-system lattice format
    '''
    fid = open(fn,'r')
    a = fid.readlines()
    fid.close()
    bl = []
    for i,le in enumerate(a[3:]):
        if verbose:
            print('LN%3d: %s'%(i+3,le))
        ele = le.split()
        if ele[1] =='DW':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'IVU':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'EPU':
            t = kmap(ele[0],L=float(ele[2]),kmap2fn=ele[7],E=3)
        elif ele[1] == 'SQ_TRIM':
            t = kick(ele[0],L=float(ele[2]))
        elif ele[1] == 'BPM':
            t = moni(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMD':
            t = kick(ele[0],L=float(ele[2]))
        elif ele[1].startwith('MARK'):
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMY':
            t = kick(ele[0],L=float(ele[2]))
        elif ele[1] == 'TRIMX':
            t = kick(ele[0],L=float(ele[2]))
        elif ele[1] == 'FTRIM':
            t = kick(ele[0],L=float(ele[2]))
        elif ele[1] == 'SEXT':
            t = sext(ele[0],L=float(ele[2]),K2=float(ele[5]))
        elif ele[1] == 'DIPOLE':
            t = bend(ele[0],L=float(ele[2]),
                     angle=float(ele[6]),e1=float(ele[6])/2,e2=float(ele[6])/2)
        elif ele[1] == 'QUAD':
            t = quad(ele[0],L=float(ele[2]),K1=float(ele[4]))
        elif ele[1].startwith('DRIF'):
            t = drif(ele[0],L=float(ele[2]))
        elif ele[1] == 'WATCH':
            t = drif(ele[0],L=float(ele[2]))
        else:
            print('unknown element type!')
            pass
        bl.append(t)
    return bl


def mad2py(fin, fout, upper=True):
    '''
    read simple/plain MAD8-like (include ELEGANT) lattice input
    format into python

    fin:  input file name (mad or elegant)
    fout: output pylatt form
    upper: if True (by default), all element names are converted
           to uppercase. Otherwise, to lowercase
    '''
    f1 = open(fin,'r')
    f2 = open(fout,'w')
    f2.write('import pylatt as latt\n\n')
    nline = 0
    while 1:
        a = f1.readline().replace(' ','').replace('\r','').upper()
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
                a = a[:-2]+f1.readline().replace(' ','').replace('\r','').upper()
                nline += 1
                print('\nLn '+str(nline)+': '+a)
            s = madParse(a[:-1],upper=upper)
            if s:
                f2.write(s+'\n')
    f2.write('ring = latt.cell(ring)')
    f1.close()
    f2.close()


def madParse(a,upper=True):
    '''
    simple mad8 parse, called by mad2py
    '''
    b = a.split(':')
    name = b[0]
    if upper:
        name = name.upper()
    else:
        name = name.lower()
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
                   'VKICK','EHKICK','EVKICK']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            else:
                print('  ignored attribute for kick: %s'%rs)
        return('%s = latt.kick(\'%s\', L = %s)'%(name,name,L))
    elif at[0] in ['KMAP','UKICKMAP']:
        L = 0
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            if 'INPUT_FILE' in rs:
                fname = rs.split('=')[1]
            else:
                print('  ignored attribute for kmap: %s'%rs)
        return('%s = latt.kmap(\'%s\', L = %s, kmap2fn = %s)'%(name,name,L,fname))
    elif at[0] in ['SCRAPER']:
        L = 0
        apxy = [-1,1,-1,1]
        for rs in at[1:]:
            if 'L=' in rs:
                L = rs.split('=')[1]
            elif 'POSITION=' in rs:
                aps = rs.split('=')[1]
            elif 'DIRECTION=' in rs:
                drt = int(rs.split('=')[1])
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
            else:
                print('  ignored attribute for scraper: %s'%rs)
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
        b = a.replace(':LINE=(',' = [').replace(')',']')
        while '-' in b:
            print('attention: beamline flip over\n')
            i = b.index('-')
            for j in xrange(i+1,len(b)):
                if b[j]==',' or b[j]==']':
                    repl = b[i+1:j]+'[::+1]'
                    break
            b = b[:i]+repl+b[j:]
        b = b.replace('+','-')
        if upper:
            return(b)
        else:
            return(b.lower())
    else:
        print('unknown type in the mad/elegant file: \'%s\''%at[0])
        return

def chkap(x,xap=[-1.,1.],yap=[-1.,1.]):
    '''
    check if particles are beyond boundary
    '''
    c1 = np.logical_and(x[0]>xap[0], x[0]<xap[1])
    c2 = np.logical_and(x[2]>yap[0], x[2]<yap[1])
    return np.logical_and(c1,c2)

def printmatrix(m,format='%9.6f',sep=' '):
    '''
    print matrix in clean format
    '''
    row,col = m.shape
    for i in range(row):
        arow = tuple([m[i,j] for j in range(col)])
        print(col*(format+sep+' '))%arow

def interp2d(x, y, z, xp, yp):
    '''
    a simple 2d linear interpolation used for ID kickmap
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
                nx2 = np.nonzero(x>=xi)[0][0]
                nx1 = nx2-1
                ny2 = np.nonzero(y>=yi)[0][0]
                ny1 = ny2-1
                x1 = x[nx1]
                x2 = x[nx2]
                y1 = y[ny1]
                y2 = y[ny2]
                f11 = z[ny1, nx1]
                f21 = z[ny1, nx2]
                f12 = z[ny2, nx1]
                f22 = z[ny2, nx2]
                zp = np.append(zp,(f11*(x2-xi)*(y2-yi)+f21*(xi-x1)*(y2-yi)+  \
                                   f12*(x2-xi)*(yi-y1)+f22*(xi-x1)*(yi-y1))/ \
                               (x2-x1)/(y2-y1))
    return zp

# --- simple optimization used for optics match
def optm(aline,var,con,wgh,xtol=1.e-8,ftol=1.e-8,
         maxiter=1000,maxfun=1000,disp=0):
    '''
    parameters optimization for beamline/cell structure
    aline: a beamline or cell
    var:   varible nested list, var[m][0] MUST be an instance
           var=[[q1,'K1'],[q2,'K1']], q1 & q2's K1 are vraible
    con:   constraint list. con[m][0] must be an attribute of beamline
           con=[['betax',0,10.0], ['emitx',5.0]], want betax to be 10m
           at beamline.s[0], and emitx 5.0nm.rad
    wgh:   weights for constraints [1,10], 1 for betax, 10 for emitx

    usage: 
    variable = [[q1,'K1'],[q2,'L']]
    constraint = [['betax',-1,10],['alfax',-1,0],['etax',-1,0],['nux',0.25]]
    weight = [1,10,10,1]
    optm(aline,variable,constraint,weight)

    optimize aline with q1.K1 and q2.L to achive betax=10, alfax=0, etax=0 
    at the end of aline, with tune is 0.25. 
    '''
    def goalfun(val,aline,var,con,wgh):
        '''
        Goal function for optimization
        val: variable value list for optimization
        aline: aline or cell/ring
        var: varible nested list, var[m][0] MUST be an instance
             var=[[q1,'K1'],[q1,'K1']], q1 & q2's K1 are vraible
        con: constraint list. con[m][0] must be an attribute of aline
             con=[['betax',0,10.0], ['emitx',5.0]], want betax to be 10m
             at aline.s[0], and emitx 5.0nm.rad
        wgh: weights for constraints [1,10], 1 for betax, 10 for emitx
        '''
        for i,v in enumerate(var):
            v[0].put(v[1],val[i])
        if aline.__class__.__name__ == 'beamline':
            aline._update()
        elif aline.__class__.__name__ == 'cell':
            fp = [a[0] for a in con]
            rlist = ['I','alphac','U0','D','Jx','Jy','Je','tau0','taux',
                     'tauy','taue','sige','emitx','sigx','sigy']
            clist = ['chx','chx0','chy','chy0']
            ra = any([a in rlist for a in fp])
            ch = any([a in clist for a in fp])
            aline.update(rad=ra,chrom=ch,verbose=False)
        if aline.__class__.__name__ == 'cell':
            if aline.isstable == 0:
                return 1e69
        gf = 0.
        for i,c in enumerate(con):
            if len(c) == 3:
                gf += wgh[i]*(c[2]-getattr(aline,c[0])[int(c[1])])**2
            else:
                gf += wgh[i]*(c[1]-getattr(aline,c[0]))**2
        return gf

    t0 = time.time()
    if len(con) != len(wgh):
        raise RuntimeError('Length of constraints and weights not matched')
    # --- get initial value
    val0 = []
    for v in var:
        val0.append(getattr(v[0],v[1]))
    val0 = np.array(val0)
    val = opt.fmin(goalfun,val0,args=(aline,var,con,wgh), \
                   xtol=xtol,ftol=ftol,maxiter=maxiter,
                   maxfun=maxfun,disp=disp)
    if disp:
        print('\nTotally %.2fs is used for optimization.\n'%(time.time()-t0))
    for i,v in enumerate(var):
        v[0].put(v[1],val[i])
        if disp:
            print(v[0])
    aline._update()
# --- end of simple optimization used for optics match


# --- four beta-function theory for linear coupling --- #
def symJ(nvar=4):
    '''
    symplectic J matrix M.T * J * M = J
    '''
    #J = np.mat(np.zeros((nvar, nvar)))
    J = np.zeros((nvar, nvar))
    for i in xrange(int(nvar/2)):
        J[2*i,2*i+1] = 1.
        J[2*i+1,2*i] = -1.
    return J

def quadrant(sn,cn):
    '''
    calculate angle from its sin and cosin
    used for phase advance
    sn: sin(angle)
    cn: cos(angle)
    return angle normalized by 2*pi
    '''
    return np.arctan2(sn,cn)/twopi

def vect2beta(v):
    '''
    use a 4x4 eigen vectors matrix to calculate all four beta-functions

    returns:
    bag: beta-alfa-gama for plane x, y and mode I, II
    bagname: name explanation for output
    ref: Willeke and Ripken, Methods of beam optics, DESY
    '''
    v = np.array(v)
    z1 = (v[:,0]+v[:,1])/np.sqrt(2)
    z2 = (v[:,0]-v[:,1])/np.sqrt(2)
    z3 = (v[:,2]+v[:,3])/np.sqrt(2)
    z4 = (v[:,2]-v[:,3])/np.sqrt(2)
    z1 = z1.real
    z2 = z2.imag
    z3 = z3.real
    z4 = z4.imag
    s = symJ(nvar=4)
    nm1 = z1.dot(s).dot(z2)
    nm2 = z3.dot(s).dot(z4)
    betax_I = (z1[0]**2+z2[0]**2)/nm1
    alfax_I = -(z1[0]*z1[1]+z2[0]*z2[1])/nm1
    gamax_I = (z1[1]**2+z2[1]**2)/nm1
    phx_I = quadrant(z2[0],z1[0])
    betay_I = (z1[2]**2+z2[2]**2)/nm1
    alfay_I = -(z1[2]*z1[3]+z2[2]*z2[3])/nm1
    gamay_I = (z1[3]**2+z2[3]**2)/nm1
    phy_I = quadrant(z2[2],z1[2])
    betax_II = (z3[0]**2+z4[0]**2)/nm2
    alfax_II = -(z3[0]*z3[1]+z4[0]*z4[1])/nm2
    gamax_II = (z3[1]**2+z4[1]**2)/nm2
    phx_II = quadrant(z4[0],z3[0])
    betay_II = (z3[2]**2+z4[2]**2)/nm2
    alfay_II = -(z3[2]*z3[3]+z4[2]*z4[3])/nm2
    gamay_II = (z3[3]**2+z4[3]**2)/nm2
    phy_II = quadrant(z4[2],z3[2])
    bagName =  np.array([['betax_I','alfax_I','gamax_I','phx_I'],
                         ['betay_I','alfay_I','gamay_I','phy_I'],\
                         ['betax_II','alfax_II','gamax_II','phx_II'],\
                         ['betay_II','alfay_II','gamay_II','phy_II']]).T
    bag =  np.array([[betax_I,alfax_I,gamax_I,phx_I],
                     [betay_I,alfay_I,gamay_I,phy_I],\
                     [betax_II,alfax_II,gamax_II,phx_II],\
                     [betay_II,alfay_II,gamay_II,phy_II]]).transpose()
    return (bag,bagName)

def monoPhase(ph):
    '''
    make phase advance be monotonic and start from zero
    '''
    n = 0
    phm = copy.deepcopy(ph)
    for i in range(1,len(phm)):
        if phm[i] < phm[i-1]:
            n += 1
            phm[i:] += 1
    phm -= phm[0]
    return phm

def tm2f2(tm,epslon=1e-6):
    '''
    convert transport matrix to linear Hamiltonian f2
    '''
    lnm2 = logm(tm) # lnm2: logarithm of linear matrix
    nd = tm.shape[0]
    S = symJ(nvar=nd)
    # cm: coefficient matrix of linear part, Hamiltonian f2 = -1/2(v.cm.v^T)
    cm = np.dot(np.linalg.inv(S),lnm2)
    xi = np.eye(nd,dtype=int)
    xv = np.ones(nd)
    xc = []
    for i in range(nd):
        xc.append(mvp.mvp(xi[i:i+1],xv[i:i+1]))
    cm = np.real(cm)
    f2 = xc[0].const(0)
    for i in range(nd):
        for j in range(nd):
            cmv = f2.const(cm[i,j])
            f2 += cmv*xc[i]*xc[j]
    f2.simplify()
    f2 = f2.const(-0.5)*f2
    f2 = f2.chop(epslon)
    return f2

def f22tm(f2,truncate=9):
    '''
    convert linear Hamiltonian f2 to transport matrix
    '''
    nd = f2.index.shape[1]
    #unit vector
    xi = np.identity(nd, int)
    xv = np.ones(nd)
    xc = []
    for i in range(nd):
        xc.append(mvp.mvp(xi[i:i+1], xv[i:i+1]))
    m = np.zeros((nd,nd))
    for ri in range(nd):
        r = f2.exp(xc[ri],truncate)
        for mi,cc in enumerate(r.index):
            ci = list(cc).index(1)
            m[ri,ci] = r.value[mi]
    return np.mat(m)

def f22tm_1(f2):
    '''
    Hamilton-Cayley, see A. Chao's lecture 9, page 21
    '''
    n = f2.index.shape[1]
    F = np.zeros((4,4))
    idx = np.zeros(n,dtype=int)
    idx[0] = 2
    F[0,0] = f2.pickWithIndex(idx)*(-2)
    idx = np.zeros(n,dtype=int)
    idx[0] = 1
    idx[1] = 1
    F[0,1] = f2.pickWithIndex(idx)*(-1)
    idx = np.zeros(n,dtype=int)
    idx[0] = 1
    idx[2] = 1
    F[0,2] = f2.pickWithIndex(idx)*(-1)
    idx = np.zeros(n,dtype=int)
    idx[0] = 1
    idx[3] = 1
    F[0,3] = f2.pickWithIndex(idx)*(-1)
    F[1,0] = F[0,1]
    idx = np.zeros(n,dtype=int)
    idx[1] = 2
    F[1,1] = f2.pickWithIndex(idx)*(-2)
    idx = np.zeros(n,dtype=int)
    idx[1] = 1
    idx[2] = 1
    F[1,2] = f2.pickWithIndex(idx)*(-1)
    idx = np.zeros(n,dtype=int)
    idx[1] = 1
    idx[3] = 1
    F[1,3] = f2.pickWithIndex(idx)*(-1)
    F[2,0] = F[0,2]
    F[2,1] = F[1,2]
    idx = np.zeros(n,dtype=int)
    idx[2] = 2
    F[2,2] = f2.pickWithIndex(idx)*(-2)
    idx = np.zeros(n,dtype=int)
    idx[2] = 1
    idx[3] = 1
    F[2,3] = f2.pickWithIndex(idx)*(-1)
    F[3,0] =F[0,3]
    F[3,1] =F[1,3]
    F[3,2] =F[2,3]
    idx = np.zeros(n,dtype=int)
    idx[3] = 2
    F[3,3] = f2.pickWithIndex(idx)*(-2)
    SF = np.dot(symJ(nvar=4),F)
    w = np.linalg.eig(SF)[0]
    a = np.ones((4,4),dtype=complex)
    a[:,1] = w
    a[:,2] = w*w
    a[:,3] = a[:,2]*w
    b = np.array([np.exp(wi) for wi in w])
    x = np.dot(np.linalg.inv(a),b)
    SF2 = np.dot(SF,SF)
    SF3 = np.dot(SF2,SF)
    M = x[0]*np.eye(4)+x[1]*SF+x[2]*SF2+x[3]*SF3
    return M.real

def f22h2(f2):
    '''
    calculate perturbative h2 (Lie map) coefficients from f2
    f2 = c * x * px ...
    h2 = c * sqrt(Jx)*exp(i*phix) ...
    bx,ax,by and ay: unperturbed optics funtion
    return h1010r,h1010i,h1001r,h1001i
    '''
    #R = f22tm(f2,truncate=5)
    R = f22tm_1(f2)
    l0,v0 = np.linalg.eig(R[:4,:4])
    bag,bagName = vect2beta(v0)
    bagx_I,bagy_I,bagx_II,bagy_II = bag[:,0],bag[:,1],bag[:,2],bag[:,3]
    bx,ax = bagx_I[0],bagx_I[1]
    by,ay = bagy_II[0],bagy_II[1]
    c1010 = f2.pickWithIndex([1,0,1,0])
    c0101 = f2.pickWithIndex([0,1,0,1])
    c0110 = f2.pickWithIndex([0,1,1,0])
    c1001 = f2.pickWithIndex([1,0,0,1])
    h1010r = c0101*(1-ax*ay)/np.sqrt(bx*by)/2 + \
             c1001*ay*np.sqrt(bx/by)/2 + \
             c0110*ax*np.sqrt(by/bx)/2 - \
             c1010*np.sqrt(bx*by)/2
    h1010i = c0101*(ax+ay)/np.sqrt(bx*by)/2 - \
             c1001*np.sqrt(bx/by)/2 - \
             c0110*np.sqrt(by/bx)/2
    h1001r = c0101*(1+ax*ay)/np.sqrt(bx*by)/2 - \
             c1001*ay*np.sqrt(bx/by)/2 - \
             c0110*ax*np.sqrt(by/bx)/2 + \
             c1010*np.sqrt(bx*by)/2
    h1001i = c0101*(ax-ay)/np.sqrt(bx*by)/2 - \
             c1001*np.sqrt(bx/by)/2 + \
             c0110*np.sqrt(by/bx)/2
    return h1010r,h1010i,h1001r,h1001i
# ---  END OF COUPLING

# --- FFT and NAFF
def pickPeak(xpw,rng=[0.,.5],ith=1):
    '''
    choose peak from fft power sprectrum within a given range
    xpw: 1d array-like sprectrum data
    rng: fractional range of peak in which a peak is being picked up
    ith: which peak to be choosen, 1st one by default
    return  peak position and peak power
    '''
    T = len(xpw)
    r0,r1 = int(rng[0]*(T-1)),int(rng[1]*(T-1))
    pk = []
    dtype = [('index',int),('value',float)]
    for i in range(r0+1,r1-1):
        if xpw[i]>xpw[i-1] and xpw[i]>xpw[i+1]:
           pk.append((i,xpw[i]))
    pk = np.sort(np.array(pk,dtype),order='value')
    if len(pk):
        return pk[-ith][0]*1./(T-1),pk[-ith][1]
    else:
        i = np.argmax(xpw[r0:r1])+r0
        return i*1./(T-1),xpw[i]


def getTuneTbT(xi,rng=[0.,0.5],ith=1,hann=False,
               verbose=False,semilog=False,
               figsize=(10,3)):
    '''
    get the fractional tune from TBT data
    normalization method provided by Weixing Cheng
    rng: the fractual range of tune range in which tune is desired
    x: 1d array-like TbT data
    ith: which peak to be choosen, 1st one by default
    return tune fractional
    '''
    x = copy.deepcopy(xi)
    x -= np.average(x)
    if not any(x):
        return 0.,0.,np.zeros((2,2)),np.zeros((2,2))
    T = len(x)
    nmf = T-1 # normalization factor
    if hann:
        hw = np.arange(T)*1.0/(T-1)
        hw = (1-np.cos(2*np.pi*hw))/2
        x *= hw
        S1 = np.sum(hw)
        S2 = np.sum(hw**2)
        nmf *= S2/S1
    xpw = np.abs(np.fft.fft(x))/nmf
    xpk = pickPeak(xpw,rng=rng,ith=ith)
    cxpw = np.cumsum(xpw)
    f0,f1 = int((T-1)*rng[0]),int((T-1)*rng[1])
    nf = np.arange(f0,f1)*1.0/(T-1)
    spt1 = np.array(zip(nf,xpw[f0:f1]))
    spt2 = np.array(zip(nf,cxpw[f0:f1]*2/cxpw[-1]))
    if verbose:
        plt.figure(figsize=figsize)
        plt.subplot(121)
        if semilog:
            plt.semilogy(spt1[:,0],spt1[:,1],'b')
        else:
            plt.plot(spt1[:,0],spt1[:,1],'b')
        plt.plot(xpk[0],xpk[1],'ro')
        plt.text(xpk[0]+0.01,xpk[1]*0.9,r'$\nu=$%.4f'%xpk[0],
                 bbox=dict(facecolor='w', alpha=1))
        plt.xlabel(r'$\nu$')
        plt.ylabel('amplitude')
        plt.subplot(122)
        if semilog:
            plt.semilogy(spt2[:,0],spt2[:,1],'b')
        else:
            plt.plot(spt2[:,0],spt2[:,1],'b')
        plt.xlabel(r'$\nu$')
        plt.ylabel(r'$\int{}\,PSD\,\mathrm{d}\nu$')
        plt.show()
    return xpk[0],xpk[1],spt1,spt2

def naffprod(n,f):
    '''
    give nu, phase, check the different between
    f: TbT data
    n: [nux,phi]
    '''
    i = np.arange(len(f))
    f1 = np.sin(np.pi*2*n[0]*i+n[1])
    a = f-sum(f*f1)/sum(f1*f1)*f1
    return sum(a*a)

def naff(f,ni=10,nran=3,verbose=0,figsize=(8,5),rng=[0,0.5]):
    '''
    NAFF: J Laskar, Physica D 56 (1992) 253-269
    f: tbt data
    ni: total number of Gramm-Schmidt orthogonalizations
    returns:
    nfnu: tune peaks
    amp: absolute amplitudes
    nran: to avoid local min, try nran times random initial phase
    '''
    fc = copy.deepcopy(f)
    fc -= np.average(fc)
    if not any(fc):
        return [0.],[0.]
    j = np.arange(len(fc))
    nfnu,amp = [],[]
    for i in xrange(ni):
        nu0 = getTuneTbT(fc,verbose=0,hann=1,rng=rng)[0]
        nu1T,fcT,zzT,reT = [],[],[],[]
        for jc in xrange(nran):
            nu = [nu0,np.random.randn()]
            nu1tmp = fmin(naffprod,nu,args=(fc,),xtol=1e-11,ftol=1e-12,disp=0)
            f1 = np.sin(np.pi*2*nu1tmp[0]*j+nu1tmp[1])
            zztmp = sum(fc*f1)/sum(f1*f1)
            fctmp = fc-zztmp*f1
            reT.append(naffprod(nu1tmp,fc))
            nu1T.append(nu1tmp)
            fcT.append(fctmp)
            zzT.append(zztmp)
        indexmin = np.argmin(reT)
        fc = fcT[indexmin]
        nfnu.append(nu1T[indexmin][0])
        amp.append(zzT[indexmin])
    if verbose:
        for i in xrange(ni):
            print('i = %2i:  nu = %10.4f, amp = %12.4e'%(i+1,nfnu[i],amp[i]))
        plt.figure(figsize=figsize)
        plt.bar(nfnu,np.abs(amp),width=1e-4)
        plt.axis([0,0.5,0,1.05*max(np.abs(amp))])
        plt.show()
    return nfnu,amp
# --- END of NAFF

def CSNormalization(betax,alfax,betay,alfay,combine=True):
    '''
    Courant Snyder normalization
    given Twiss, return normalization matrix
    if combine is true, return x-ip transfer matrix
    '''
    sqtbx = np.sqrt(betax)
    sqtby = np.sqrt(betay)
    m = np.mat([[1/sqtbx,0,0,0],[alfax/sqtbx,sqtbx,0,0],
                [0,0,1/sqtby,0],[0,0,alfay/sqtby,sqtby]])
    if not combine:
        return m
    else:
        cs = np.mat([[1,-1j,0,0],[0,0,1,-1j]])
        return cs*m


# --- Li-Hua Yu matrix method
def coefb0(bL,betax,phix,mux):
    """
    Copy from Li-Hua Yu
    bL: list-like K2L/2 for sextupole
    betax: list-like beta-x at the locations of sexts
    phix: list-like phase-x at the locations of sexts
    mux: tune*2*pi
    all lists should have a same length
    """
    ns = len(betax)

    #term ~ hxp^3 in 3rd order terms of b0
    b03z312=sum( [-bL[i]*bL[j]*np.sqrt(betax[i]**3*betax[j]**3)\
    *np.exp(-1j*phix[i]-1j*phix[j])* ( np.exp(2j*phix[j]) +np.exp(2j*phix[i]))*\
    ( np.exp(1j*mux+2j*phix[j]) + np.exp(2j*phix[i]) ) \
    for i in range(ns) for j in range(ns) if i>j ] \
    )\
    /8./(-1+np.exp(1j*mux))**2/(1+np.exp(1j*mux))

    b03z322=sum( [-bL[i]**2*betax[i]**3*np.exp(2j*phix[i])for i in range(ns) ]) \
    /8./(-1+np.exp(1j*mux))**2

    b03z3=b03z312+b03z322

    #term ~ hxm^3 in 3rd order terms of b0
    b03zs312=sum( [bL[i]*bL[j]*np.sqrt(betax[i]**3*betax[j]**3)*\
    ( \
    np.exp(4j*mux-1j*phix[j]-3j*phix[i])*(-1-np.exp(1j*mux)-np.exp(2j*mux)+np.exp(3j*mux))\
    +np.exp(4j*mux-3j*phix[j]-1j*phix[i])*(1-np.exp(1j*mux)-np.exp(2j*mux)-np.exp(3j*mux)) \
    ) for i in range(ns) for j in range(ns) if i>j ]) \
    /8./(1-np.exp(3j*mux)-np.exp(4j*mux)+np.exp(7j*mux))

    b03zs322=sum( [-bL[i]**2*betax[i]**3*np.exp(5j*mux-4j*phix[i]) for i in range(ns)  ])\
    /8./(-1+np.exp(1j*mux))**2/(1+np.exp(1j*mux)+2*np.exp(2j*mux)+np.exp(3j*mux)+np.exp(4j*mux))

    b03zs3=b03zs312+b03zs322

    #term ~ hxm^2*hxp in 3rd order terms of b0
    b03zs2z12=sum( [bL[i]*bL[j]*np.sqrt(betax[i]**3*betax[j]**3)*\
    (\
    -np.exp(2j*mux-1j*phix[j]-1j*phix[i])*(1+np.exp(1j*mux))*(1+np.exp(1j*mux)+np.exp(2j*mux))\
    + np.exp(2j*mux-3j*phix[j]+1j*phix[i])*(3+np.exp(1j*mux)+np.exp(2j*mux)) \
    + np.exp(3j*mux+1j*phix[j]-3j*phix[i])*(1+np.exp(1j*mux)+3*np.exp(2j*mux))
    )\
    for i in range(ns) for j in range(ns) if i>j ] \
    ) \
    /8./(1-np.exp(2j*mux)-np.exp(3j*mux)+np.exp(5j*mux))

    b03zs2z22=sum( [bL[i]**2*betax[i]**3*np.exp(2j*mux-2j*phix[i]) for i in range(ns)  ])\
    *(1+np.exp(3j*mux)) /8./(1-np.exp(2j*mux)-np.exp(3j*mux)+np.exp(5j*mux))

    b03zs2z=b03zs2z12+b03zs2z22

    #term ~ hxm^2 in 2rd order terms of b0
    b02zs2=1j*sum( [bL[i]*np.sqrt(betax[i]**3)*np.exp(3j*mux-3j*phix[i]) for i in range(ns)  ])\
    /4./(-1+np.exp(3j*mux))

    #term ~ hxp^2 in 2rd order terms of b0
    b02z2=-1j*sum( [bL[i]*np.sqrt(betax[i]**3)*np.exp(1j*phix[i]) for i in range(ns)  ])\
    /4./(-1+np.exp(1j*mux))

    #term ~ hxm*hxp in 2rd order terms of b0
    b02zsz=-1j*sum( [bL[i]*np.sqrt(betax[i]**3)*np.exp(1j*mux-1j*phix[i]) for i in range(ns)  ])\
    /2./(1-np.exp(1j*mux))

    return b02z2,b02zs2,b02zsz,b03z3,b03zs3,b03zs2z

def b0z(xbar,pbar,b02z2,b02zs2,b02zsz,b03z3,b03zs3,b03zs2z):
    z = xbar-1j*pbar
    zs = np.conj(z)
    b01 = z
    b02 = b02z2*z**2+b02zs2*zs**2+b02zsz*z*zs
    b03 = b03z3*z**3+b03zs3*zs**3+b03zs2z*zs**2*z
    return abs((b01+b02+b03))
# --- Li-Hua Yu matrix method
