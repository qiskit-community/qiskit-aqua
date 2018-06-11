#
# Class QCOpMat for operator matrices.
#

"""
A class for individual operator matrices from/to a matrix-element file.
Module qcmatrixio provides low-level I/O routines for the file.

Class
-----

OpMat -- one operator matrix from/to a matrix element file.

Properties
----------

name    -- name string (label used in file)

type    -- "i", "d", or "c" for integer (int32), double, or
           complex double.

asym    -- whether any lower-triangular indices are antisymmetric/anti-Hermetian
           (True) or symmetric/Hermetian (False).

nelem   -- number of elements per set of indicies.  Usually 1, but
           2 or 3 for raffenetti 1,2 or 1,2, and 3 integrals.

dimens   -- tuple with up to 5 dimensions, negative values for lower-triangular
            tetrahedral, etc. storage.

array    -- array with the actual data.  A linear np.ndarray unless
            nelem>1, in which case it is (ntot,nelem) (fortran order).

lenarray -- the number of sets of nelem values in array,so
            array.size=self.lenarry*self.nelem

labpars  -- a tuple of values for a header record for the object in a file.

Methods
-------

OpMat (name,array,nelem=1,type=None,asym=False,dimens=None):
  initialization, copies arguments to the corresponding properties.
  type defaults based on the data type in the array, which must be
  np.int32, np.int64, np.float64, or np.complex128.  asym is False for
  symmetric/Hermetian and True for antisymmetric/anti-Hermetian
  and only matters if dimens marks some some indices as lower
  triangular/tetrahedral/etc.  dimens defaults to one dimension
  determined by the size of the array (i.e., array.size/nelem)
  
print_mat (wid=1,**kwargs):
  print the matrix; wid selects different line lengths and formats.
  this is also invoked indirectly when the __str__ method is used.

(ind,sign) = make_indxf (*args):
  return an index into the array for a given set of arguments and
  which the sign should be flipped (or complex conjugate taken).
  This takes fortran-style indices -- starting at 1 with the
  leftmost fastest running
  
(ind,sign) = make_indxc (*args):
  return an index into the array for a given set of arguments and
  which the sign should be flipped (or complex conjugate taken).
  This takes c/python-style indices -- starting at 0 with the
  rightmost fastest running.

get_elemf (*args):
set_elemf (value,*args):
  return/set-and-return an element of the array given by a list of
  fortran-style indices.

get_elemc (*args):
set_elemc (value,*args):
  return/set-and-return an element of the array given by a list of
  c-style indices.

expand:
  expand the array in self from lower-triangular to full within
  the object and returns the resulting array.

wr_lbuf (iu,lenbuf):
  write the object to a matrix element file open on fortran unit iu
  using lenbuf as the max buffer (record) size.

wr_lrind (iu,lenbuf):
  write an object containing a real array in compressed form, with
  indices for the non-zero elements

wr_lao2e (iu,lenbuf):
  write an object containing AO 2e integrals, compressed with 4
  indices for each non-zero integral (or Raff set).

write(iu,lenbuf):
  write an object in either 2e or uncompressed form as appropriate.

Other functions
---------------

Most functions which accept optional keyword arguments pass these
on to print(), but can also include the keyword input=True to
print in the form of an executable statement.

printlab (cbuf,ni,nr,nri,ntot,lenbuf,n1,n2,n3,n4,n5,asym,**kwargs)
  print a line giving the parameters for a matrix.

def doinpprt (label,x,doinp=False,**kwargs):
  print array x with label as an executable statement.

def print1d (comp,type,wid,label,arr,**kwargs):
  print a 1-dimensional array.  comp is True to compress,
  printing only non-zero values.

def print2e (cbuf,nbasis,r,**kwargs):
  print an array of two-electron integrals with label cbuf

def ltout (label,n,x,key,im,**kwargs):
  lower-triangular matrix output, im is non-zero to print
  a submatrix number.

def sqout (label,m,n,x,key,im,**kwargs):
  square matrix output, im is non-zero to print
 a submatrix number.

"""

__version__ = 2.0
import sys
import io
import re
import numpy as np
import os
INTSIZE_NAME = "GAUOPEN_INTSIZE"
doi8 = False
import qcmatrixio as qcmio

INPKW = "input"

def _lenarray (d):
  if len(d) == 5:
    l = d[4]*qcmio.lind4(False,d[0],d[1],d[2],d[3],0,abs(d[0]),abs(d[1]),abs(d[2]),abs(d[3]))[0] + 1
  elif len(d) == 4:
    l = qcmio.lind4(False,d[0],d[1],d[2],d[3],0,abs(d[0]),abs(d[1]),abs(d[2]),abs(d[3]))[0] + 1
  elif len(d) == 3:
    l = qcmio.lind3(False,d[0],d[1],d[2],0,abs(d[0]),abs(d[1]),abs(d[2]))[0] + 1
  elif len(d) == 2:
    l = qcmio.lind2(False,d[0],d[1],0,abs(d[0]),abs(d[1]))[0] + 1
  else:
    l = d[0]
  return (l)

def _makeindx (dimens,asym,args):
  if len(dimens) >= 5: return qcmio.lind5 (True,dimens[0],dimens[1],dimens[2],dimens[3],
                                        dimens[4],asym,args[0],args[1],args[2],args[3],args[4])
  elif len(dimens) == 4: return qcmio.lind4 (True,dimens[0],dimens[1],dimens[2],dimens[3],
                                          asym,args[0],args[1],args[2],args[3])
  elif len(dimens) == 3: return qcmio.lind3 (True,dimens[0],dimens[1],dimens[2],asym,args[0],
                                          args[1],args[2])
  elif len(dimens) == 2: return qcmio.lind2 (True,dimens[0],dimens[1],asym,args[0],args[1])
  else:  return (args[0]-1,1.0e0)

def _makeindxc (dimens,asym,args):
  if len(dimens) >= 5: return qcmio.lind5 (True,dimens[0],dimens[1],dimens[2],dimens[3],
                                           dimens[4],asym,args[4]+1,args[3]+1,args[2]+1,
                                           args[1]+1,args[0]+1)
  elif len(dimens) == 4: return qcmio.lind4 (True,dimens[0],dimens[1],dimens[2],dimens[3],
                                             asym,args[3]+1,args[2]+1,args[1]+1,args[0]+1)
  elif len(dimens) == 3: return qcmio.lind3 (True,dimens[0],dimens[1],dimens[2],asym,
                                             args[2]+1,args[1]+1,args[0]+1)
  elif len(dimens) == 2: return qcmio.lind2 (True,dimens[0],dimens[1],asym,args[1]+1,args[0]+1)
  else:  return (args[0],1.0e0)

def printlab (cbuf,ni,nr,nri,ntot,lenbuf,n1,n2,n3,n4,n5,asym,doinp=False,**kwargs):
  if doinp: print(cbuf," = ",end=" ",**kwargs)
  else:
    if asym: iasym = -1
    else: iasym = 0
    print ("%-35s NI=%2d NR=%2d NRI=%1d NTot=%8d" % (cbuf,ni,nr,nri,ntot),end=" ",**kwargs)
    if lenbuf > 0: print ("LenBuf=%8d" % lenbuf,end=" ",**kwargs)
    print ("N=%6d%6d%6d%6d%6d AS=%2d" % (n1,n2,n3,n4,n5,iasym),**kwargs)

def formatv (fwid,plusstr,pkstr,thresh,val):
  if abs(val) < thresh: val1 = 0.0e0
  else: val1 = val
  str = (pkstr % val1).strip()
  if re.match("^-0\.0*$",str): str = str[1:]
  if str[0] != '-': str = plusstr + str
  str = str.replace('e','D')
  return (fwid % str)

def formatx (fwid,plusstr,pkstr,thresh,val):
  if type(val) == np.complex128 or type(val) == complex:
    str1 = formatv (fwid," ",pkstr,0.0e0,val.real)
    str2 = formatv (fwid,"+",pkstr,0.0e0,val.imag)
    str = str1 + str2 + "i "
  else:
    str = formatv(fwid,plusstr,pkstr,thresh,val)
  return(str)

def printpars (type,wid):
  if (type == "i"):
    if (wid == 1):
      npl = 20
      pkstr = "%4d"
      fwid = "%4s"
    elif (wid == 2):
      npl = 10
      pkstr = "%8d"
      fwid = "%8s"
    else:
      npl = 5
      pkstr = "%12d"
      fwid = "%12s"
  elif (type == "d"):
    if (wid == 1):
      npl = 10
      pkstr = "%12.6f"
      fwid = "%12s"
    elif (wid == 2):
      npl = 3
      pkstr = "%12.6f"
      fwid = "%12s"
    elif (wid == 3):
      npl = 5
      pkstr = "%20.8f"
      fwid = "%20s"
    else:
      npl = 5
      pkstr = "%12.6f"
      fwid = "%12s"
  elif (type == "c"):
    npl = 5
    pkstr = "%12.6f"
    fwid = "%12s"
  else:
    print ("error",**kwargs)
    npl = 1
    pkstr = "%12.6f"
    fwid = "%12s"
    raise TypeError
  return (npl,pkstr,fwid)

def doinpprt (label,x,doinp=False,**kwargs):
  if doinp:
    if label != " ": print("  elif name == \"%s\":\n    arr = np." % label,end="",**kwargs)
    np.set_printoptions (threshold=1000000000)
    xstr = x.__repr__()
    print (xstr,**kwargs)
  return (doinp)

def print1d (comp,type,wid,label,arr,doinp=False,**kwargs):
  if arr is None: return
  if doinpprt (label,arr,doinp=False,**kwargs): return
  if (label != " "): labstr = "%6s=" % label
  else: labstr = "  "
  npl,pkstr,fwid = printpars (type,wid)
  i = 0
  ndone = 0
  while (i < arr.size):
    if (abs(arr[i]) >= 1.e-12) or not comp:
      if ndone == 0: print (labstr,end="",**kwargs)
      if comp:  print("%8d=" % (i+1),end="",**kwargs)
      str = formatx (fwid," ",pkstr,0.0e0,arr[i])
      print (str,end="",**kwargs)
      ndone = ndone + 1
      if ndone == npl:
        print ("",**kwargs)
        ndone = 0
    i = i + 1
  if ndone > 0: print ("",**kwargs)

def print2e (cbuf,n,r,doinp=False,**kwargs):
  if doinpprt (cbuf,r,doinp=False,**kwargs): return
  if re.match("^REG",cbuf): lab = "Int"
  else: lab = "R1"
  if len(np.shape(r)) == 1:
    lr = np.shape(r)
    nr = 1
  else: lr,nr = np.shape(r)
  ri = np.empty((nr))
  for i in range(n):
    for j in range(i+1):
      for k in range(i+1):
        if i == k: llim = j + 1
        else: llim = k + 1
        for l in range(llim):
          ijkl,sign = qcmio.lind4(False,-n,-n,-n,n,0,i+1,j+1,k+1,l+1)
          doit = False
          if (nr == 1):
            ri[0] = r[ijkl]
            doit = doit or (abs(ri[0]) >= 1.e-12)
          else:
            for x in range(nr):
              ri[x] = r[ijkl,x]
              doit = doit or (abs(ri[x]) >= 1.e-12)
          if doit:
            str = "I=%3i J=%3i K=%3i L=%3i %s=%20.12e" % (i+1,j+1,k+1,l+1,lab,ri[0])
            if nr > 1:  str = str + " R2=%20.12e" % ri[1]
            if nr > 2:  str = str + " R3=%20.12e" % ri[2]
            str = str.replace("e","D")
            print (str,**kwargs)

def ltout (label,n,x,key,im,doinp=False,**kwargs):
  if doinpprt (label,x,doinp=False,**kwargs): return
  if key > 0: thresh = 0.0e0
  else:  thresh = 10.0e0**(key-6)
  ntt = (n*(n+1))//2
  if im > 0:
    print ("%s, matrix %6d:" % (label,im),**kwargs)
    imoff = (im-1)*ntt
  else: imoff = 0
  if (type(x[0]) == np.complex128):
    nc = 4
    fmthead = "%19i           "
  else:
    nc = 5
    fmthead = "%14i"
  for ist in range(0,n,nc):
    iend = min(ist+nc,n)
    for irow in range (ist,iend): print (fmthead % (irow+1),end="",**kwargs)
    print (**kwargs)
    for irow in range (ist,n):
      ir = min(irow-ist+1,nc)
      l = (irow*(irow+1))//2 + ist + imoff
      print ("%4d" % (irow+1),end="",**kwargs)
      for i in range(ir):
        s = x[l]
        l = l + 1
        s = formatx ("%14s","","%14.6e",thresh,s)
        print (s,end="",**kwargs)
      print (**kwargs)

def sqout (label,m,n,x,key,im,doinp=False,**kwargs):
  if doinpprt (label,x,doinp=False,**kwargs): return
  if key > 0: thresh = 0.0e0
  else:  thresh = 10.0e0**(key-6)
  if im > 0:
    print ("%s, matrix %6d:" % (label,im),**kwargs)
    imoff = (im-1)*m*n
  else: imoff = 0
  if (type(x[0]) == np.complex128):
    nc = 4
    fmthead = "%23i       "
    fmtval = "%14.6e"
  elif (type(x[0]) == np.float64):
    nc = 5
    fmthead = "%14i"
    fmtval = "%14.6e"
  else:
    nc = 5
    fmthead = "%14i"
    fmtval = "%14d"
  for jl in range(0,n,nc):
    ju = min(jl+nc,n)
    num = ju - jl
    for j in range (jl,ju):  print (fmthead % (j+1),end="",**kwargs)
    print (**kwargs)
    for i in range(m):
      imx = i + imoff
      print ("%7d " % (i+1),end="",**kwargs)
      for j in range(jl,ju):
        s = formatx ("%14s","",fmtval,thresh,x[imx+j*m])
        print (s,end="",**kwargs)
      print (**kwargs)

class OpMat (object):

  def __init__ (self,name,array,nelem=1,type=None,asym=False,dimens=None):
    if isinstance (name,str): self.name = name
    else: raise TypeError
    if isinstance (array,np.ndarray): self.array = array
    else: raise TypeError
    if isinstance (nelem,int): self.nelem = nelem
    else: raise TypeError
    if type is None:
      if self.array.dtype == np.int32: self.type = "i"
      elif self.array.dtype == np.int64: self.type = "i"
      elif self.array.dtype == np.float64: self.type = "d"
      elif self.array.dtype == np.complex128: self.type = "c"
      else: raise TypeError
    elif not isinstance (type,str): raise TypeError
    else: self.type = type
    if asym: self.asym = True
    else: self.asym = False
    if dimens is None: self.dimens = (self.array.size/self.nelem,)
    elif not isinstance (dimens,tuple): raise TypeError
    else: self.dimens = dimens

  @property
  def lenarray (self):
    return _lenarray (self.dimens)

  @property
  def labpars (self):
    if (self.type == "c"):
      ni = 0
      nr = self.nelem
      nri = 2
    elif (self.type == "d"):
      ni = 0
      nr = self.nelem
      nri = 1
    else:
      ni = self.nelem
      nr = 0
      nri = 1
    ntot = self.lenarray
    n1 = self.dimens[0]
    n2 = self.dimens[1] if len(self.dimens) >= 2 else 1
    n3 = self.dimens[2] if len(self.dimens) >= 3 else 1  
    n4 = self.dimens[3] if len(self.dimens) >= 4 else 1  
    n5 = self.dimens[3] if len(self.dimens) >= 5 else 1  
    return (self.name,ni,nr,nri,ntot,n1,n2,n3,n4,n5,self.asym)

  def print_mat (self,wid=1,doinp=False,**kwargs):
    name,ni,nr,nri,ntot,n1,n2,n3,n4,n5,asym = self.labpars
    if doinpprt (name,self.array,doinp=False,**kwargs): return
    printlab (name,ni,nr,nri,ntot,0,n1,n2,n3,n4,n5,asym,**kwargs)
    if qcmio.aoints(name): print2e (self.name,self.dimens[3],self.array,**kwargs)
    elif len(self.dimens) == 1:
      if re.match("GAUSSIAN SCALARS",self.name): print1d (True,self.type,5," ",self.array,**kwargs)
      else: print1d (False,self.type,wid," ",self.array,**kwargs)
    elif len(self.dimens) == 2:
      if self.dimens[0] < 0: ltout (" ",self.dimens[1],self.array,0,0,**kwargs)
      else: sqout (" ",self.dimens[0],self.dimens[1],self.array,0,0,**kwargs)
    elif len(self.dimens) >= 3:
      allpos = True
      nmat = 1
      for i in range(2,len(self.dimens)):
        nmat = nmat * self.dimens[i]
        allpos = allpos and (self.dimens[i] > 0)
      if self.dimens[0] < 0 and self.dimens[1] > 0 and allpos:
        for im in range(nmat): ltout(name,self.dimens[1],self.array,0,im+1,**kwargs)
      elif self.dimens[0] > 0 and self.dimens[1] > 0 and allpos:
        for im in range(self.dimens[2]): sqout(name,self.dimens[0],self.dimens[1],self.array,0,im+1,**kwargs)
      elif (len(self.dimens) >= 4) and (self.dimens[0] == -self.dimens[1]) and (self.dimens[2] == -self.dimens[3]):
        nmat = (self.dimens[3]*(self.dimens[3]+1))//2
        if len(self.dimens) >= 5: nmat = self.dimens[4]*nmat
        for im in range(nmat): ltout(name,self.dimens[1],self.array,0,im+1,**kwargs)
      else: print1d (False,self.type,1," ",self.array,**kwargs)
    else:
      print ("cannot print dims",self.dimens,**kwargs)

  def __str__ (self):
    stream = io.StringIO()
    self.print_mat(file=stream)
    str = stream.getvalue()
    return (str[:-1])

  def make_indxf (self,*args):
    return _makeindx(self.dimens,self.asym,args)
  
  def make_indxc (self,*args):
    return _makeindxc(self.dimens,self.asym,args)

  def get_elemf (self,*args):
    indx,sign = _makeindx(self.dimens,self.asym,args)
    val = self.array[indx]
    if sign < 0:
      if self.type == "c": val = val.conjugate()
      if self.asym: val = -val
    return val

  def get_elemc (self,*args):
    indx,sign = _makeindxc(self.dimens,self.asym,args)
    val = self.array[indx]
    if sign < 0:
      if self.type == "c": val = val.conjugate()
      if self.asym: val = -val
    return val
  
  def set_elemf (self,value,*args):
    indx,sign = _makeindx(self.dimens,self.asym,args)
    val = value
    if sign < 0:
      if self.type == "c": val = val.conjugate()
      if self.asym: val = -val
    self.array[indx] = val
    return self.array[indx]

  def set_elemc (self,value,*args):
    indx,sign = _makeindxc(self.dimens,self.asym,args)
    val = value
    if sign < 0:
      if self.type == "c": val = val.conjugate()
      if self.asym: val = -val
    self.array[indx] = val
    return self.array[indx]
  
  def expand (self):
    d = tuple(reversed([abs(num) for num in self.dimens]))
    if qcmio.aoints(self.name):
      if self.dimens[0] < 0:
        n = self.dimens[3]
        lr = self.array.size//self.nelem
        if self.nelem == 1: narr = qcmio.expao1(n,self.array)
        else: narr = qcmio.expaon(n,self.array)
      else: narr = self.array
    else:
      narr = np.empty(d,dtype=type(self.array[0]))
      for i in np.ndindex(*d): narr[i] = self.get_elemc(*i)
    self.array = narr.reshape((_lenarray(d)))
    self.dimens = tuple(reversed(d))
    return (self.array)

  def wr_lbuf(self,iu,lenbuf):
    label,ni,nr,nri,ntot,n1,n2,n3,n4,n5,asym = self.labpars
    lenbx = lenbuf - (lenbuf % (nri * self.nelem))
    lenbx = lenbx//nri
    qcmio.wr_labl(iu,label,ni,nr,ntot,lenbx,n1,n2,n3,n4,n5,asym)
    if self.type == "i": qcmio.wr_ibuf(iu,lenbx,self.array)
    elif self.type == "c": qcmio.wr_cbuf(iu,lenbx,self.array)
    else: qcmio.wr_rbuf(iu,lenbx,self.array)

  def wr_lrind (iu,lenbuf):
    ntot = self.lenarr
    lenbx = lenbuf//self.nelem
    y = self.array.reshape((self.nelem,ntot),order='F')
    nnz = qcmio.numnzr(y)
    wr_labl(iu,self.name,1,nr,nnz,lenbx,ntot,1,1,1,1,0)
    wr_rind(iu,nnz,lenbx,y)

  def wr_lao2e (self,iu,lenbuf):
    label,ni,nr,nri,ntot,n1,n2,n3,n4,n5,asym = self.labpars
    ntot = self.lenarray
    if ((ntot*self.nelem) != self.array.size) or (self.nelem > 3):
      print ("2e write error NTot=",ntot,"nelem=",self.nelem,"size",self.array.size)
      raise TypeError
    lenbx = lenbuf//(2+self.nelem)
    nnz = qcmio.numnza(self.array)
    qcmio.wr_labl(iu,label,4,nr,nnz,lenbx,n1,n2,n3,n4,n5,asym)
    qcmio.wr_2e(iu,nnz,self.dimens[3],lenbx,self.array)

  def write(self,iu,lenbuf):
    if qcmio.aoints(self.name): self.wr_lao2e(iu,lenbuf)
    else: self.wr_lbuf(iu,lenbuf)
