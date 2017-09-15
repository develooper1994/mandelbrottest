
# coding: utf-8

# In[21]:

#biraz da görsellik ama burada performansla ilgilenmedim sadece şeklin aslını görün, neyle uğraşıyoruz.
#pylab çalışması için pyqt4 yüklü olmak zorunda
from pylab import *
from numpy import NaN
 
def m(a):
    z = 0
    for n in range(1, 100):
        z = z**2 + a
        if abs(z) > 2:
            return n
    return NaN
 
X = arange(-2, .5, .002)
Y = arange(-1,  1, .002)
Z = zeros((len(Y), len(X)))
 
for iy, y in enumerate(Y):
    print (iy, "of", len(Y))
    for ix, x in enumerate(X):
        Z[iy,ix] = m(x + 1j * y)
 
imshow(Z, cmap = plt.cm.prism, interpolation = 'none', extent = (X.min(), X.max(), Y.min(), Y.max()))
xlabel("Re(c)")
ylabel("Im(c)")
savefig("mandelbrot_python.png")
show()


# In[1]:

#sadece python ile
import math

def mandelbrot(z, c, n=40):
    if abs(z) > 1000:
        return float("nan")
    elif n > 0:
        return mandelbrot(z ** 2 + c, c, n - 1)
    else:
        return z ** 2 + c
print("\n".join(["".join(["#" if not math.isnan(mandelbrot(0, x + 1j * y).real) else " "                           for x in [a * 0.02 for a in range(-80, 30)]])                  for y in [a * 0.05 for a in range(-20, 20)]])       )


# In[9]:

#birçok dilde olan reduction algoritmasıyla mandelbrot
try:
    from functools import reduce
except:
    pass
 
 
def mandelbrot(a):
    return reduce(lambda z, _: z * z + a, range(50), 0)
 
def step(start, step, iterations):
    return (start + (i * step) for i in range(iterations))
 
rows = (("*" if abs(mandelbrot(complex(x, y))) < 2 else " "
        for x in step(-2.0, .0315, 80))
        for y in step(1, -.05, 41))
 
print("\n".join("".join(row) for row in rows))


# In[16]:

#timeit modülünün nasıl çalıştığını göstermek için
import timeit
mandelbrot_timeit='''
import time
import numpy as np

def mandelbrot2(z,maxiter):
    c = z
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1,r2,[mandelbrot2(complex(r, i),maxiter) for r in r1 for i in r2])
mandelbrot_set(-2.0,0.5,-1.25,1.25,1000,1000,80)
'''
print('timeit aslında küçük fonksiyonları daha pythonu açmadan denemediz için dizayn edilmiştir. timet it sonucu {}'.format(
      timeit.timeit(stmt=mandelbrot_timeit)))
#not: uzun fonksiyonlar için çooooooook uzun sürüyor. uzun fonksiyonlar için clock kullanın.
#hata var çünkü işlem bitmedi ve işlemi kestim.


# In[6]:

#diğer bir yol,numpy kullandım. bu performans ölçmeyi göstermek için
import time
import numpy as np

def mandelbrot2(z,maxiter):
    c = z
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return maxiter

def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1,r2,[mandelbrot2(complex(r, i),maxiter) for r in r1 for i in r2])

t0 = time.clock()
mandelbrot_set(-2.0,0.5,-1.25,1.25,1000,1000,80)
print(time.clock()-t0)

t0 = time.clock()
mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)


# In[31]:

import time
import numpy as np
import numexpr as ne

def mandelbrot_numpy2(c, maxiter):
    output = np.zeros(c.shape)
    z = np.zeros(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = np.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = it
        z[notdone] = z[notdone]**2 + c[notdone]
    output[output == maxiter-1] = 0
    return output

def mandelbrot_set2(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy2(c,maxiter)
    return (r1,r2,n3.T) 

t0 = time.clock()
mandelbrot_set2(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)
#numexpr olmdadan sadece numpy ile 131.5 saniye de hesaplamıştı şimdi 23.4 saniye . yine tek iyi gelişme ama numba kadar değil.


# In[17]:

#anaconda içinde bulunan numba paketiyle jit(just in time yani o anda derle) yöntemiyle hız elde etmeyi planladım.
#temelde fazla birşey değişmiyor. sadece yukarıya bir jit geliyor ve çıkışı return yerine n3 içindede depolayıp veriyorum.
#numba sizin python ile işlemcinin erişemediğiniz özelliklerine erişme imkanı verir. numbayı sevin ve öğrenin.

from numba import jit
import numpy as np
import time

@jit
def mandelbrot(c,maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return 0

@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j],maxiter)
    return (r1,r2,n3)

t0 = time.clock()
mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)
#DİKKAT 2048 İLE BİTEN 121.5 SANİYEDEN 4.5 saniyeye indi ve bundan sonra da 2048 kulanıldı.


# In[19]:

#daha iyi optimizasyonla biraz daha hızlandırma belki?
@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return 0


@jit
def mandelbrot_set4(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i],r2[j],maxiter)
    return (r1,r2,n3)

t0 = time.clock()
mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)
#4.945943999421161-4.118945251433161=0.826998747988000 yani 0.83 saniye hızlandı ama yetmez biraz da gösrsellik.


# In[76]:

# numepr numbada vektörleştirilmiş mandelbrot. hadi biraz daha hız ver bana. inanıyorum sana.
#yemek tarifi gibi oldu :)
from numba import vectorize, complex64, boolean, jit
import numpy as np
import time
import numexpr as ne

@vectorize([boolean(complex64)])
def f(z):
    return (z.real*z.real + z.imag*z.imag) < 4.0

@vectorize([complex64(complex64, complex64)])
def g(z,c):
    return z*z + c 

@jit
def mandelbrot_numpy(c, maxiter):
    output = np.zeros(c.shape, np.int)
    z = np.empty(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = f(z)
        output[notdone] = it
        z[notdone] = g(z[notdone],c[notdone]) 
    output[output == maxiter-1] = 0
    return output
#yeni mandelbrot_numpy kullanıyor ama daha yavaş yani OLMADI
def mandelbrot_set3(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    n3 = mandelbrot_numpy(c,maxiter)
    n3 = n3.reshape((width,height))
    return (r1,r2,n3.T)


@jit
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return 0
#eski mandelbrot kullanıyor ama daha hızlı
def mandelbrot_set5(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i],r2[j],maxiter)
    return (r1,r2,n3)

#eski
#işte böyle 2.7 saniye. zamanla yarışıyoruz burda.
t0 = time.clock()
mandelbrot_set5(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)
#yeni
t0 = time.clock()
mandelbrot_set3(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)


# In[77]:

from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32

@jit(int32(complex64, int32))
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

def mandelbrot_set_guvectorize(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
    return (r1,r2,n3.T) 

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i],maxiter)
        
def mandelbrot_set_numpy_guvectorize(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
    return (r1,r2,n3.T) 

#gpu olmadan jit INT32
t0 = time.clock()
mandelbrot_set_guvectorize(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)

#yeni mandelbrot_numpy versiyonu yeni adam oldu ama gpu üzerinde.
t0 = time.clock()
mandelbrot_set_numpy_guvectorize(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)


# In[79]:

#guvectorized cuda üzerinde

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),(n)->(n)', target='cuda')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        creal = c[i].real
        cimag = c[i].imag
        real = creal
        imag = cimag
        output[i] = 0
        for n in range(maxiter):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                output[i] = n
                break
            imag = 2* real*imag + cimag
            real = real2 - imag2 + creal
            
def mandelbrot_set_guvectorize_cuda(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = np.empty(c.shape, int)
    maxit = np.ones(c.shape, int) * maxiter
    n3 = mandelbrot_numpy(c,maxit)
    return (r1,r2,n3.T) 

t0 = time.clock()
mandelbrot_set_guvectorize_cuda(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
print(time.clock()-t0)


# In[ ]:



